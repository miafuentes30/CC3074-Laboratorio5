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
