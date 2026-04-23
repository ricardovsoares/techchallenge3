"""
config.py
=========
Centraliza todas as configurações do projeto: paths, hiperparâmetros,
listas de features e constantes de domínio.

Decisão de design: manter tudo em um único arquivo facilita reprodutibilidade
e evita problemas espalhados pelo código.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "flights.csv"

# ─── Coluna alvo ──────────────────────────────────────────────────────────────
# Classificação: voo atrasou >= DELAY_THRESHOLD minutos na chegada?
TARGET_CLASSIFICATION = "DELAYED"
TARGET_REGRESSION = "ARRIVAL_DELAY"
DELAY_THRESHOLD = 15   # minutos — padrão FAA para definir "atraso"

# ─── Features utilizadas nos modelos ──────────────────────────────────────────
# Excluímos colunas que são "vazar informação do futuro" (leakage):
#   ARRIVAL_TIME, DEPARTURE_TIME, WHEELS_OFF, WHEELS_ON, etc.
CATEGORICAL_FEATURES = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "DAY_OF_WEEK",
    "MONTH",
]

NUMERICAL_FEATURES = [
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "DEPARTURE_DELAY",   # atraso na partida — principal preditor do atraso na chegada
    "TAXI_OUT",
    "DAY",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# ─── Features para clusterização (sem leakage) ────────────────────────────────
CLUSTER_FEATURES = [
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "MONTH",
    "DAY_OF_WEEK",
]

# ─── Amostragem ───────────────────────────────────────────────────────────────
# Dataset original ~5M linhas; usamos amostra para prototipagem rápida.
# Setar SAMPLE_FRAC=1.0 para treinar no dataset completo.
SAMPLE_FRAC = 0.05   # 5% ≈ 250k linhas
RANDOM_STATE = 42

# ─── Split treino/teste ───────────────────────────────────────────────────────
TEST_SIZE = 0.2

# ─── Hiperparâmetros default ──────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 50,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# Regressão
RF_REG_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 50,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LGB_REG_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# ─── Clusterização ────────────────────────────────────────────────────────────
N_CLUSTERS_RANGE = range(2, 11)   # intervalo para o Elbow Method
OPTIMAL_K = 4              # definido após análise do Elbow + Silhouette

# ─── PCA ──────────────────────────────────────────────────────────────────────
PCA_VARIANCE_THRESHOLD = 0.95   # manter componentes que explicam 95% da variância

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
