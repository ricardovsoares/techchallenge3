"""
data_preprocessing.py
=====================
Módulo responsável por carregar, limpar e preparar os dados brutos de voos.

Decisões de design:
- Leitura com tipos explícitos para economizar memória (~40% menos RAM).
- Remoção de voos cancelados/desviados antes da análise de atrasos,
  pois esses registros não possuem ARRIVAL_DELAY válido.
- Imputação da mediana para variáveis numéricas (robusta a outliers).
- Imputação com moda/categoria "UNKNOWN" para categóricas.
- Engenharia de features derivadas: período do dia, estação do ano,
  rota combinada (ORIGIN→DEST) e flag de horário de pico.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    from config import (
        RAW_DATA_PATH, SAMPLE_FRAC, RANDOM_STATE,
        TARGET_CLASSIFICATION, TARGET_REGRESSION, DELAY_THRESHOLD,
    )
except ImportError:
    from src.config import (
        RAW_DATA_PATH, SAMPLE_FRAC, RANDOM_STATE,
        TARGET_CLASSIFICATION, TARGET_REGRESSION, DELAY_THRESHOLD,
    )


# ─── Dtypes otimizados ────────────────────────────────────────────────────────
DTYPE_MAP = {
    "YEAR":               "int16",
    "MONTH":              "int8",
    "DAY":                "int8",
    "DAY_OF_WEEK":        "int8",
    "AIRLINE":            "category",
    "FLIGHT_NUMBER":      "int32",
    "ORIGIN_AIRPORT":     "category",
    "DESTINATION_AIRPORT":"category",
    "SCHEDULED_DEPARTURE":"int16",
    "DEPARTURE_TIME":     "float32",
    "DEPARTURE_DELAY":    "float32",
    "TAXI_OUT":           "float32",
    "WHEELS_OFF":         "float32",
    "SCHEDULED_TIME":     "float32",
    "ELAPSED_TIME":       "float32",
    "AIR_TIME":           "float32",
    "DISTANCE":           "float32",
    "WHEELS_ON":          "float32",
    "TAXI_IN":            "float32",
    "SCHEDULED_ARRIVAL":  "int16",
    "ARRIVAL_TIME":       "float32",
    "ARRIVAL_DELAY":      "float32",
    "DIVERTED":           "int8",
    "CANCELLED":          "int8",
    "CANCELLATION_REASON":"category",
    "AIR_SYSTEM_DELAY":   "float32",
    "SECURITY_DELAY":     "float32",
    "AIRLINE_DELAY":      "float32",
    "LATE_AIRCRAFT_DELAY":"float32",
    "WEATHER_DELAY":      "float32",
}


def load_raw(path: Path = RAW_DATA_PATH, sample_frac: float = SAMPLE_FRAC) -> pd.DataFrame:
    """
    Lê o CSV com dtypes otimizados e aplica amostragem estratificada por mês
    para preservar a distribuição temporal.
    """
    log.info(f"Lendo {path} …")
    df = pd.read_csv(path, dtype={k: v for k, v in DTYPE_MAP.items()
                                  if k not in ("CANCELLATION_REASON",)},
                     low_memory=False)
    log.info(f"Shape original: {df.shape}")

    if sample_frac < 1.0:
        df = (df.groupby("MONTH")
                .sample(frac=sample_frac, random_state=RANDOM_STATE))
        log.info(f"Shape após amostragem ({sample_frac*100:.0f}%): {df.shape}")

    return df.reset_index(drop=True)


def remove_cancelled_diverted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove voos cancelados e desviados.
    Decisão: esses voos não possuem ARRIVAL_DELAY real e introduziriam
    ruído nos modelos de classificação/regressão de atrasos.
    """
    before = len(df)
    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)].copy()
    log.info(f"Removidos cancelados/desviados: {before - len(df):,} linhas")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estratégia de imputação:
    - Numérico: mediana (robusta a outliers de atraso).
    - Categórico: moda ou 'UNKNOWN' quando moda não existe.
    - Colunas de delay de causa (AIR_SYSTEM_DELAY etc.): preenchidas com 0
      pois NaN indica ausência do tipo de atraso, não dado faltante.
    """
    delay_cause_cols = [
        "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
        "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
    ]
    for col in delay_cause_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include="category").columns.tolist()
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else "UNKNOWN"
            df[col] = df[col].cat.add_categories(["UNKNOWN"]) if "UNKNOWN" not in df[col].cat.categories else df[col]
            df[col] = df[col].fillna(fill_val)

    log.info("Imputação concluída.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis derivadas para enriquecer o modelo:

    - PERIOD_OF_DAY : madrugada / manhã / tarde / noite com base em SCHEDULED_DEPARTURE
    - SEASON        : estação do ano (1=Inverno … 4=Outono) com base em MONTH
    - ROUTE         : par ORIGIN→DESTINATION como feature categórica de rota
    - IS_PEAK_HOUR  : flag se voo parte entre 7-9h ou 17-19h (horário de pico)
    - DEP_HOUR      : hora da partida programada (0-23)
    """
    df = df.copy()

    # Hora da partida programada
    df["DEP_HOUR"] = df["SCHEDULED_DEPARTURE"] // 100

    # Período do dia
    conditions = [
        df["DEP_HOUR"].between(0, 5),
        df["DEP_HOUR"].between(6, 11),
        df["DEP_HOUR"].between(12, 17),
        df["DEP_HOUR"].between(18, 23),
    ]
    choices = ["MADRUGADA", "MANHA", "TARDE", "NOITE"]
    df["PERIOD_OF_DAY"] = np.select(conditions, choices, default="MANHA")

    # Estação do ano (hemisfério norte, base dos dados EUA)
    season_map = {
        12: "INVERNO", 1: "INVERNO",  2: "INVERNO",
         3: "PRIMAVERA", 4: "PRIMAVERA", 5: "PRIMAVERA",
         6: "VERAO",    7: "VERAO",    8: "VERAO",
         9: "OUTONO",  10: "OUTONO",  11: "OUTONO",
    }
    df["SEASON"] = df["MONTH"].map(season_map)

    # Rota
    df["ROUTE"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)

    # Horário de pico
    df["IS_PEAK_HOUR"] = df["DEP_HOUR"].isin(list(range(7, 10)) + list(range(17, 20))).astype(int)

    # Converter novas categóricas
    for col in ["PERIOD_OF_DAY", "SEASON", "ROUTE"]:
        df[col] = df[col].astype("category")

    log.info("Engenharia de features concluída.")
    return df


def create_target(df: pd.DataFrame, threshold: int = DELAY_THRESHOLD) -> pd.DataFrame:
    """
    Cria a variável alvo binária DELAYED:
      1 se ARRIVAL_DELAY >= threshold (padrão 15 min)
      0 caso contrário (incluindo chegadas adiantadas)
    """
    df = df.copy()
    df[TARGET_CLASSIFICATION] = (df[TARGET_REGRESSION] >= threshold).astype(int)
    log.info(f"Taxa de atraso: {df[TARGET_CLASSIFICATION].mean():.2%}")
    return df


def preprocess_pipeline(path: Path = RAW_DATA_PATH,
                         sample_frac: float = SAMPLE_FRAC) -> pd.DataFrame:
    """
    Pipeline completo: carrega → limpa → imputa → engenharia → cria alvo.
    Retorna DataFrame pronto para EDA e modelagem.
    """
    df = load_raw(path, sample_frac)
    df = remove_cancelled_diverted(df)
    df = impute_missing(df)
    df = engineer_features(df)
    df = create_target(df)
    log.info(f"Shape final do pipeline: {df.shape}")
    return df


if __name__ == "__main__":
    df = preprocess_pipeline()
    print(df[["AIRLINE", "ORIGIN_AIRPORT", "ARRIVAL_DELAY", "DELAYED",
              "PERIOD_OF_DAY", "SEASON"]].head())
    print(df.dtypes)
