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


