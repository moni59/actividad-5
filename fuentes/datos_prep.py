"""Carga, exporta y prepara Breast Cancer Wisconsin Diagnostic."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datos" / "datos_ini" / "breast_cancer_wisconsin.csv"
CLEAN = ROOT / "datos" / "datos_limp" / "breast_cancer_wisconsin_limpio.csv"
TARGET = "diagnostico"


def cargar_original():
    """Retorna el conjunto incorporado en scikit-learn; 0=maligno, 1=benigno."""
    ds = load_breast_cancer(as_frame=True)
    df = ds.data.copy()
    df[TARGET] = ds.target.astype("int8")
    df.index.name = "id_muestra"
    return df


def validar_y_limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres/tipos y elimina filas duplicadas; no imputa ni escala aquí."""
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    out = out.drop_duplicates().reset_index(drop=True)
    out[TARGET] = pd.to_numeric(out[TARGET], errors="raise").astype("int8")
    features = out.drop(columns=TARGET)
    out[features.columns] = features.apply(pd.to_numeric, errors="raise")
    if out.isna().any().any():
        raise ValueError("Se encontraron valores nulos inesperados")
    if not set(out[TARGET].unique()).issubset({0, 1}):
        raise ValueError("La etiqueta debe ser binaria (0/1)")
    return out


def exportar():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    original = cargar_original()
    original.to_csv(RAW, index=True)
    limpio = validar_y_limpiar(original)
    limpio.to_csv(CLEAN, index=False)
    return limpio


if __name__ == "__main__":
    df = exportar()
    print(f"Datos limpios: {df.shape[0]} filas × {df.shape[1]} columnas -> {CLEAN}")
