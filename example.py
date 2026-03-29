"""
CC3074 - Minería de Datos
Laboratorio 5: Naive Bayes
Integrantes:
  - Mia Alejandra Fuentes Merida, 23775
  - María José Girón Isidro, 23559
  - Leonardo Dufrey Mejía Mejía, 23648

"""

# IMPORTS
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass  
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import pyreadr
except ModuleNotFoundError:
    pyreadr = None

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

plt.rcParams["figure.figsize"] = (12, 7)
plt.style.use("ggplot")

_plot_counter = 0

def display_plot():
    """Muestra gráfica interactivamente o la guarda como PNG en modo no interactivo."""
    global _plot_counter
    if "agg" in plt.get_backend().lower():
        _plot_counter += 1
        filename = f"./plots/grafica_{_plot_counter:03d}.png"
        import os
        os.makedirs("./plots", exist_ok=True)
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# CARGA DE DATOS (LAB 4)
RDATA_PATH = "./data/listings.RData"
CSV_PATH   = "./data/collectedData.csv"

def load_csv_fallback(csv_path, required_col="price", chunksize=100_000):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return None
    header = None
    used_enc = None
    for enc in ("utf-8", "latin-1"):
        try:
            header = pd.read_csv(csv_file, nrows=0, encoding=enc)
            used_enc = enc
            break
        except UnicodeDecodeError:
            continue
    if header is None or required_col not in header.columns:
        return None
    frames = []
    for chunk in pd.read_csv(csv_file, encoding=used_enc,
                             low_memory=True, chunksize=chunksize):
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else None

df_raw = None
if pyreadr is not None:
    try:
        resultado_r = pyreadr.read_r(RDATA_PATH)
        if resultado_r:
            df_raw = next(iter(resultado_r.values()))
            print(f"Datos cargados desde {RDATA_PATH}")
    except Exception as e:
        print(f"Error leyendo RData: {e}")

if df_raw is None:
    df_raw = load_csv_fallback(CSV_PATH)
    if df_raw is not None:
        print(f"Datos cargados desde {CSV_PATH}")

if df_raw is None:
    print("No se pudieron cargar los datos.")
    sys.exit(1)

# PREPROCESAMIENTO (LAB 4)
df = df_raw.copy()

df["price"] = (
    df["price"].astype(str)
    .str.replace(r"[$,]", "", regex=True).str.strip()
    .replace("", np.nan)
)
df["price"] = pd.to_numeric(df["price"], errors="coerce")

for col in ["host_response_rate", "host_acceptance_rate"]:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace("%", "", regex=False).str.strip()
            .replace("", np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

bool_cols = [
    c for c in df.columns
    if str(df[c].dtype) in ("object", "string")
    and df[c].dropna().isin(["t", "f"]).all()
]
for col in bool_cols:
    df[col] = df[col].map({"t": 1, "f": 0})

drop_cols = [c for c in df.columns if any(
    kw in c.lower() for kw in [
        "url", "description", "name", "summary", "space",
        "neighborhood_overview", "notes", "transit", "access",
        "interaction", "house_rules", "thumbnail", "medium",
        "picture", "xl_picture", "host_about", "scrape_id",
        "last_scraped", "calendar_last_scraped", "license"
    ]
)]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

for col in ["bathrooms", "bedrooms", "beds"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

df = df[(df["price"] > 0) & (df["price"] <= 1000)].copy()
df.dropna(subset=["price"], inplace=True)

print(f"Dataset limpio: {df.shape}")

# VARIABLE CATEGORICA (LAB 4)
q33 = df["price"].quantile(0.33)
q66 = df["price"].quantile(0.66)
print(f"P33 = ${q33:.2f}  |  P66 = ${q66:.2f}")

def clasificar_precio(p):
    if p <= q33:   return "Economica"
    elif p <= q66: return "Intermedia"
    else:           return "Cara"

df["precio_cat"] = df["price"].apply(clasificar_precio)

# MATRICES X/Y REGRESION
df_num = df.select_dtypes(include=[np.number]).copy()
drop_eda = [c for c in ["capacity_group", "review_group"] if c in df_num.columns]
df_num.drop(columns=drop_eda, inplace=True, errors="ignore")

y = df_num["price"].copy()
X = df_num.drop(columns=["price"]).copy()
X = X.dropna(axis=1, how="all")
X = X.fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nREGRESIÓN — Train: {X_train.shape} | Test: {X_test.shape}")

# MATRICES X/Y CLASIFICACION
df_cls = df_num.copy()
df_cls["precio_cat"] = df["precio_cat"].values
df_cls.drop(columns=["price"], inplace=True, errors="ignore")

X_cls = df_cls.drop(columns=["precio_cat"]).copy()
X_cls = X_cls.dropna(axis=1, how="all")
X_cls = X_cls.fillna(X_cls.median(numeric_only=True))
y_cls = df_cls["precio_cat"]

X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)
print(f"CLASIFICACIÓN — Train: {X_cls_train.shape} | Test: {X_cls_test.shape}")


# ACTIVIDAD 1 - NB REGRESION (BASE)
print("\n" + "="*60)
print("ACTIVIDAD 1 — NB REGRESIÓN (modelo base)")
print("="*60)

pipeline_nb_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("nb",     GaussianNB())
])

t0 = time.time()
pipeline_nb_reg.fit(X_train, y_train)
tiempo_nb_reg = time.time() - t0

y_pred_nb_reg = pipeline_nb_reg.predict(X_test)

mse_nb_reg  = mean_squared_error(y_test, y_pred_nb_reg)
rmse_nb_reg = np.sqrt(mse_nb_reg)
r2_nb_reg   = r2_score(y_test, y_pred_nb_reg)
mae_nb_reg  = mean_absolute_error(y_test, y_pred_nb_reg)

print(f"RMSE : {rmse_nb_reg:.4f}")
print(f"R²   : {r2_nb_reg:.4f}")
print(f"MAE  : {mae_nb_reg:.4f}")
print(f"Tiempo entrenamiento: {tiempo_nb_reg:.3f}s")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_nb_reg, alpha=0.25, color="steelblue", s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Precio real (USD)")
plt.ylabel("Precio predicho (USD)")
plt.title(f"NB Regresión Base | R²={r2_nb_reg:.3f}  RMSE={rmse_nb_reg:.2f}")
plt.tight_layout()
display_plot()

errores = y_pred_nb_reg - y_test
plt.figure(figsize=(8, 4))
sns.histplot(errores, kde=True, color="coral")
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Diferencia (Predicción - Real)")
plt.ylabel("Frecuencia")
plt.title("Distribución de errores — NB Regresión")
plt.tight_layout()
display_plot()


# ACTIVIDAD 2 - ANALISIS NB REGRESION
print("\n" + "="*60)
print("ACTIVIDAD 2 — ANÁLISIS NB REGRESIÓN")
print("="*60)
print(f"RMSE: {rmse_nb_reg:.4f} | R2: {r2_nb_reg:.4f} | MAE: {mae_nb_reg:.4f}")


# ACTIVIDAD 3 - COMPARACION REGRESION
print("\n" + "="*60)
print("ACTIVIDAD 3 — COMPARACIÓN: REGRESIÓN LINEAL vs ÁRBOL vs NB")
print("="*60)

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
pipeline_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  RidgeCV(alphas=alphas, cv=5,
                       scoring="neg_root_mean_squared_error"))
])
t0 = time.time()
pipeline_ridge.fit(X_train, y_train)
tiempo_ridge = time.time() - t0
y_pred_ridge = pipeline_ridge.predict(X_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge   = r2_score(y_test, y_pred_ridge)

t0 = time.time()
arbol_reg = DecisionTreeRegressor(max_depth=10, random_state=42)
arbol_reg.fit(X_train, y_train)
tiempo_arbol = time.time() - t0
y_pred_arbol = arbol_reg.predict(X_test)
rmse_arbol = np.sqrt(mean_squared_error(y_test, y_pred_arbol))
r2_arbol   = r2_score(y_test, y_pred_arbol)

t0 = time.time()
rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_train)
tiempo_rf_reg = time.time() - t0
y_pred_rf_reg = rf_reg.predict(X_test)
rmse_rf_reg = np.sqrt(mean_squared_error(y_test, y_pred_rf_reg))
r2_rf_reg   = r2_score(y_test, y_pred_rf_reg)

comp_reg = pd.DataFrame({
    "Modelo":  [
        "Regresión Lineal (Ridge)",
        "Árbol Regresión (d=10)",
        "Random Forest Regresión (n=200, d=12)",
        "Naive Bayes"
    ],
    "RMSE":    [
        round(rmse_ridge, 4),
        round(rmse_arbol, 4),
        round(rmse_rf_reg, 4),
        round(rmse_nb_reg, 4)
    ],
    "R²":      [
        round(r2_ridge, 4),
        round(r2_arbol, 4),
        round(r2_rf_reg, 4),
        round(r2_nb_reg, 4)
    ],
    "Tiempo(s)":[
        round(tiempo_ridge, 3),
        round(tiempo_arbol, 3),
        round(tiempo_rf_reg, 3),
        round(tiempo_nb_reg, 3)
    ]
})
print("\nComparación modelos de regresión (mismo X_train/X_test):")
print(comp_reg.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colores = ["#DD8452", "#4C72B0", "#2ca02c"]
axes[0].bar(comp_reg["Modelo"], comp_reg["RMSE"], color=colores)
axes[0].set_title("RMSE (menor = mejor)")
axes[0].set_ylabel("RMSE")
axes[0].tick_params(axis="x", rotation=12)
axes[1].bar(comp_reg["Modelo"], comp_reg["R²"], color=colores)
axes[1].set_title("R² (mayor = mejor)")
axes[1].set_ylabel("R²")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=12)
plt.suptitle("Comparación modelos de regresión — Lab 4 + NB", fontsize=13, fontweight="bold")
plt.tight_layout()
display_plot()

mejor_modelo_reg = comp_reg.loc[comp_reg["R²"].idxmax(), "Modelo"]
print(f"\nMejor modelo de regresión: {mejor_modelo_reg}")


# ACTIVIDAD 4 - NB CLASIFICACION (BASE)
print("\n" + "="*60)
print("ACTIVIDAD 4 — NB CLASIFICACIÓN (modelo base)")
print("="*60)

pipeline_nb_cls = Pipeline([
    ("scaler", StandardScaler()),
    ("nb",     GaussianNB())
])

t0 = time.time()
pipeline_nb_cls.fit(X_cls_train, y_cls_train)
tiempo_nb_cls = time.time() - t0

print(f"Modelo entrenado en {tiempo_nb_cls:.3f}s")
print(f"Clases: {pipeline_nb_cls.named_steps['nb'].classes_}")


# ACTIVIDAD 5 - EFICIENCIA EN PRUEBA
print("\n" + "="*60)
print("ACTIVIDAD 5 — EFICIENCIA EN CONJUNTO DE PRUEBA")
print("="*60)

y_cls_pred_nb = pipeline_nb_cls.predict(X_cls_test)

acc_nb_cls = accuracy_score(y_cls_test, y_cls_pred_nb)
print(f"Accuracy NB Clasificación: {acc_nb_cls:.4f} ({acc_nb_cls*100:.2f}%)")
print("\nReporte de clasificación:")
print(classification_report(
    y_cls_test, y_cls_pred_nb,
    target_names=["Cara", "Economica", "Intermedia"]
))


# ACTIVIDAD 6 - MATRIZ DE CONFUSION
print("\n" + "="*60)
print("ACTIVIDAD 6 — MATRIZ DE CONFUSIÓN NB CLASIFICACIÓN")
print("="*60)

LABELS = ["Economica", "Intermedia", "Cara"]
cm_nb = confusion_matrix(y_cls_test, y_cls_pred_nb, labels=LABELS)

disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=LABELS)
fig, ax = plt.subplots(figsize=(7, 6))
disp_nb.plot(ax=ax, colorbar=True, cmap="Blues")
plt.title("Matriz de Confusión — Naive Bayes Clasificación")
plt.tight_layout()
display_plot()

print("\nMatriz de confusión detallada:")
df_cm = pd.DataFrame(
    cm_nb,
    index=[f"Real: {l}" for l in LABELS],
    columns=[f"Pred: {l}" for l in LABELS]
)
print(df_cm)


# ACTIVIDAD 7 - SOBREAJUSTE
print("\n" + "="*60)
print("ACTIVIDAD 7 — ANÁLISIS DE SOBREAJUSTE")
print("="*60)

acc_nb_train = accuracy_score(
    y_cls_train, pipeline_nb_cls.predict(X_cls_train)
)
print(f"Accuracy en ENTRENAMIENTO: {acc_nb_train:.4f}")
print(f"Accuracy en PRUEBA:        {acc_nb_cls:.4f}")
print(f"Diferencia (train-test):   {acc_nb_train - acc_nb_cls:+.4f}")


# ACTIVIDAD 8 - VALIDACION CRUZADA
print("\n" + "="*60)
print("ACTIVIDAD 8 — VALIDACIÓN CRUZADA (10-fold)")
print("="*60)

cv_nb_cls = cross_val_score(
    pipeline_nb_cls, X_cls_train, y_cls_train,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring="accuracy"
)

print(f"Accuracy por fold: {np.round(cv_nb_cls, 4)}")
print(f"Accuracy promedio CV: {cv_nb_cls.mean():.4f}")
print(f"Desviación estándar:  {cv_nb_cls.std():.4f}")
print(f"Accuracy en prueba:   {acc_nb_cls:.4f}")

plt.figure(figsize=(6, 4))
plt.boxplot(cv_nb_cls, patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.7))
plt.axhline(acc_nb_cls, color="red", linestyle="--",
            label=f"Acc Test = {acc_nb_cls:.4f}")
plt.title("Distribución Accuracy — Validación Cruzada 10-fold")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
display_plot()


# ACTIVIDAD 9A - TUNEO NB CLASIFICACION
print("\n" + "="*60)
print("ACTIVIDAD 9A — TUNEO NB CLASIFICACIÓN (var_smoothing)")
print("="*60)

param_grid_cls = {
    "nb__var_smoothing": np.logspace(0, -9, num=50)
}

grid_nb_cls = GridSearchCV(
    pipeline_nb_cls,
    param_grid=param_grid_cls,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1
)
grid_nb_cls.fit(X_cls_train, y_cls_train)

mejor_vs_cls = grid_nb_cls.best_params_["nb__var_smoothing"]
print(f"Mejor var_smoothing (clasificación): {mejor_vs_cls:.2e}")
print(f"Mejor accuracy CV: {grid_nb_cls.best_score_:.4f}")

y_pred_nb_cls_tuned = grid_nb_cls.best_estimator_.predict(X_cls_test)
acc_nb_cls_tuned = accuracy_score(y_cls_test, y_pred_nb_cls_tuned)
tiempo_nb_cls_tuned = grid_nb_cls.refit_time_
print(f"Accuracy en prueba (tuned): {acc_nb_cls_tuned:.4f}")
print(f"Accuracy en prueba (base):  {acc_nb_cls:.4f}")
print(f"Mejora: {acc_nb_cls_tuned - acc_nb_cls:+.4f}")

results_cls = pd.DataFrame(grid_nb_cls.cv_results_)
results_cls["var_smoothing"] = results_cls["param_nb__var_smoothing"]
results_cls["accuracy_mean"] = results_cls["mean_test_score"]

plt.figure(figsize=(8, 4))
plt.plot(results_cls["var_smoothing"], results_cls["accuracy_mean"],
         color="steelblue")
plt.xscale("log")
plt.xlabel("var_smoothing")
plt.ylabel("Accuracy (CV)")
plt.title("Accuracy vs var_smoothing — NB Clasificación")
plt.axvline(mejor_vs_cls, color="red", linestyle="--",
            label=f"Mejor = {mejor_vs_cls:.2e}")
plt.legend()
plt.tight_layout()
display_plot()

# ACTIVIDAD 9B - TUNEO NB REGRESION
print("\n" + "="*60)
print("ACTIVIDAD 9B — TUNEO NB REGRESIÓN (var_smoothing)")
print("="*60)

from sklearn.model_selection import KFold

max_tune_samples = 20_000
if len(X_train) > max_tune_samples:
    idx_tune = X_train.sample(n=max_tune_samples, random_state=42).index
    X_tune_reg = X_train.loc[idx_tune]
    y_tune_reg = y_train.loc[idx_tune]
else:
    X_tune_reg = X_train
    y_tune_reg = y_train

var_smoothing_vals = np.logspace(0, -9, num=25)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_cv_por_vs = []

for vs in var_smoothing_vals:
    pipe_tmp = Pipeline([
        ("scaler", StandardScaler()),
        ("nb",     GaussianNB(var_smoothing=vs))
    ])
    scores = -cross_val_score(
        pipe_tmp, X_tune_reg, y_tune_reg,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=1
    )
    rmse_cv_por_vs.append(scores.mean())

idx_mejor = int(np.argmin(rmse_cv_por_vs))
mejor_vs_reg  = var_smoothing_vals[idx_mejor]
mejor_rmse_cv = rmse_cv_por_vs[idx_mejor]

print(f"Mejor var_smoothing (regresión): {mejor_vs_reg:.2e}")
print(f"Mejor RMSE CV: {mejor_rmse_cv:.4f}")

pipeline_nb_reg_tuned = Pipeline([
    ("scaler", StandardScaler()),
    ("nb",     GaussianNB(var_smoothing=mejor_vs_reg))
])
pipeline_nb_reg_tuned.fit(X_train, y_train)
y_pred_nb_reg_tuned = pipeline_nb_reg_tuned.predict(X_test)
rmse_nb_reg_tuned = np.sqrt(mean_squared_error(y_test, y_pred_nb_reg_tuned))
r2_nb_reg_tuned   = r2_score(y_test, y_pred_nb_reg_tuned)

print(f"RMSE en prueba (tuned): {rmse_nb_reg_tuned:.4f}")
print(f"RMSE en prueba (base):  {rmse_nb_reg:.4f}")
print(f"R² (tuned): {r2_nb_reg_tuned:.4f}")
print(f"R² (base):  {r2_nb_reg:.4f}")
print(f"Mejora RMSE: {rmse_nb_reg - rmse_nb_reg_tuned:+.4f}")

plt.figure(figsize=(8, 4))
plt.plot(var_smoothing_vals, rmse_cv_por_vs, color="coral")
plt.xscale("log")
plt.xlabel("var_smoothing")
plt.ylabel("RMSE (CV 5-fold)")
plt.title("RMSE vs var_smoothing — NB Regresión")
plt.axvline(mejor_vs_reg, color="red", linestyle="--",
            label=f"Mejor = {mejor_vs_reg:.2e}")
plt.legend()
plt.tight_layout()
display_plot()


# ACTIVIDAD 10 - COMPARACION CLASIFICACION FINAL
print("\n" + "="*60)
print("ACTIVIDAD 10 — COMPARACIÓN CLASIFICACIÓN FINAL")
print("="*60)

t0 = time.time()
arbol_cls = DecisionTreeClassifier(max_depth=12, random_state=42)
arbol_cls.fit(X_cls_train, y_cls_train)
tiempo_arbol_cls = time.time() - t0
acc_arbol_cls = accuracy_score(y_cls_test, arbol_cls.predict(X_cls_test))

t0 = time.time()
rf_cls = RandomForestClassifier(n_estimators=100, max_depth=10,
                                 random_state=42, n_jobs=-1)
rf_cls.fit(X_cls_train, y_cls_train)
tiempo_rf_cls = time.time() - t0
acc_rf_cls = accuracy_score(y_cls_test, rf_cls.predict(X_cls_test))

comp_cls = pd.DataFrame({
    "Modelo":     ["Árbol Clasificación (d=12)",
                   "Random Forest (n=100, d=10)",
                   "Naive Bayes (base)",
                   "Naive Bayes (tuned)"],
    "Accuracy":   [round(acc_arbol_cls, 4),
                   round(acc_rf_cls, 4),
                   round(acc_nb_cls, 4),
                   round(acc_nb_cls_tuned, 4)],
    "Tiempo(s)":  [round(tiempo_arbol_cls, 3),
                   round(tiempo_rf_cls, 3),
                   round(tiempo_nb_cls, 3),
                   round(tiempo_nb_cls_tuned, 3)]
})
print("\nComparación modelos de clasificación (mismo split):")
print(comp_cls.to_string(index=False))

pred_arbol_cls = arbol_cls.predict(X_cls_test)
pred_rf_cls = rf_cls.predict(X_cls_test)

cm_arbol = confusion_matrix(y_cls_test, pred_arbol_cls, labels=LABELS)
cm_rf = confusion_matrix(y_cls_test, pred_rf_cls, labels=LABELS)
cm_nb_tuned = confusion_matrix(y_cls_test, y_pred_nb_cls_tuned, labels=LABELS)

print("\nMatriz de confusión — Árbol de Clasificación:")
print(pd.DataFrame(
    cm_arbol,
    index=[f"Real: {l}" for l in LABELS],
    columns=[f"Pred: {l}" for l in LABELS]
))

print("\nMatriz de confusión — Random Forest:")
print(pd.DataFrame(
    cm_rf,
    index=[f"Real: {l}" for l in LABELS],
    columns=[f"Pred: {l}" for l in LABELS]
))

print("\nMatriz de confusión — Naive Bayes (tuned):")
print(pd.DataFrame(
    cm_nb_tuned,
    index=[f"Real: {l}" for l in LABELS],
    columns=[f"Pred: {l}" for l in LABELS]
))

plt.figure(figsize=(9, 5))
colores = ["#4C72B0", "#2ca02c", "#DD8452", "#9467bd"]
bars = plt.bar(comp_cls["Modelo"], comp_cls["Accuracy"], color=colores)
for bar, acc in zip(bars, comp_cls["Accuracy"]):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.003,
             f"{acc:.2%}", ha="center", va="bottom", fontsize=10)
plt.title("Comparación de Accuracy — Clasificación de Precio")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=12)
plt.tight_layout()
display_plot()

plt.figure(figsize=(9, 4))
bars_t = plt.bar(comp_cls["Modelo"], comp_cls["Tiempo(s)"], color=colores)
for bar, t in zip(bars_t, comp_cls["Tiempo(s)"]):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{t:.3f}s", ha="center", va="bottom", fontsize=10)
plt.title("Tiempo de Entrenamiento — Comparación de Modelos")
plt.ylabel("Segundos")
plt.xticks(rotation=12)
plt.tight_layout()
display_plot()

mejor_modelo_cls = comp_cls.loc[comp_cls["Accuracy"].idxmax(), "Modelo"]
mas_lento = comp_cls.loc[comp_cls["Tiempo(s)"].idxmax(), "Modelo"]
print(f"\nMejor modelo de clasificación (accuracy): {mejor_modelo_cls}")
print(f"Modelo más lento: {mas_lento}")

# COMPARACION FINAL - REGRESION
print("\n" + "="*60)
print("COMPARACIÓN FINAL — REGRESIÓN (todos los modelos)")
print("="*60)

comp_reg_final = pd.DataFrame({
    "Modelo":   ["Regresión Lineal (Ridge)",
                 "Árbol Regresión (d=10)",
                 "Random Forest Regresión (n=200, d=12)",
                 "Naive Bayes (base)",
                 "Naive Bayes (tuned)"],
    "RMSE":     [round(rmse_ridge, 4),
                 round(rmse_arbol, 4),
                 round(rmse_rf_reg, 4),
                 round(rmse_nb_reg, 4),
                 round(rmse_nb_reg_tuned, 4)],
    "R²":       [round(r2_ridge, 4),
                 round(r2_arbol, 4),
                 round(r2_rf_reg, 4),
                 round(r2_nb_reg, 4),
                 round(r2_nb_reg_tuned, 4)]
})
print(comp_reg_final.to_string(index=False))

mejor_reg_final = comp_reg_final.loc[comp_reg_final["R²"].idxmax(), "Modelo"]
print(f"\nMejor modelo de regresión hasta el momento: {mejor_reg_final}")

print("\n" + "="*60)
print("Lab 5 completado :)")
print("="*60)

