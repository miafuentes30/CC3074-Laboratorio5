# CC3074-Laboratorio5

## Requisitos

- Windows + PowerShell
- Python 3.12 o 3.13

## Dependencias


- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- pyreadr

## Instalacion (PowerShell)

Desde la carpeta del proyecto:

```powershell
# 1) Crear entorno virtual (si no existe)
python -m venv .venv

# 2) Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# 3) Actualizar herramientas base
python -m pip install --upgrade pip setuptools wheel

# 4) Instalar dependencias del proyecto
python -m pip install numpy pandas matplotlib seaborn scikit-learn pyreadr
```

## Ejecutar

```powershell
python .\main.py
```

