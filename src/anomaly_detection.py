"""
anomaly_detection.py
====================
Detecção de anomalias em voos — identificação de rotas e companhias
com padrões atípicos de atraso.

Decisões de design:
───────────────────
- Isolation Forest: escolhido por eficiência em datasets grandes e por
  não assumir distribuição normal (atrasos são altamente assimétricos).
- Local Outlier Factor (LOF): complementar ao Isolation Forest; detecta
  anomalias locais (pontos incomuns em relação aos vizinhos).
- Threshold de contaminação: 5% (proporção esperada de anomalias).
  Ajustável via parâmetro `contamination`.
- Features usadas: DEPARTURE_DELAY, ARRIVAL_DELAY, TAXI_OUT, TAXI_IN,
  SCHEDULED_TIME — captura voos incomuns em múltiplas dimensões.
- Output: DataFrame com flag `IS_ANOMALY` + visualização dos anomalias
  em espaço bidimensional (redução PCA).
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

try:
    from config import OUTPUT_DIR, RANDOM_STATE
except ImportError:
    from src.config import OUTPUT_DIR, RANDOM_STATE

ANOM_DIR = OUTPUT_DIR / "anomaly"
ANOM_DIR.mkdir(parents=True, exist_ok=True)

ANOMALY_FEATURES = [
    "DEPARTURE_DELAY", "ARRIVAL_DELAY",
    "TAXI_OUT", "TAXI_IN", "SCHEDULED_TIME",
]
CONTAMINATION = 0.05  # 5% de anomalias esperadas


def prepare_anomaly_data(df: pd.DataFrame, features=None) -> tuple:
    if features is None:
        features = ANOMALY_FEATURES
    features = [f for f in features if f in df.columns]
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X.index, features


def run_isolation_forest(X_scaled: np.ndarray,
                          contamination: float = CONTAMINATION) -> np.ndarray:
    """
    Isolation Forest: isola pontos anomalosos por partições aleatórias.
    Retorna array de labels: -1 = anomalia, 1 = normal.
    """
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    labels = iso.fit_predict(X_scaled)
    n_anomalies = (labels == -1).sum()
    log.info(f"Isolation Forest: {n_anomalies:,} anomalias detectadas ({n_anomalies/len(labels):.1%})")
    return labels


def run_lof(X_scaled: np.ndarray,
            contamination: float = CONTAMINATION) -> np.ndarray:
    """
    Local Outlier Factor: detecta pontos com densidade local muito baixa
    em relação aos vizinhos.
    """
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        n_jobs=-1,
    )
    labels = lof.fit_predict(X_scaled)
    n_anomalies = (labels == -1).sum()
    log.info(f"LOF: {n_anomalies:,} anomalias detectadas ({n_anomalies/len(labels):.1%})")
    return labels


def plot_anomalies(X_scaled: np.ndarray,
                   iso_labels: np.ndarray,
                   lof_labels: np.ndarray):
    """
    Reduz para 2D via PCA e plota pontos normais vs anomalias
    para cada método lado a lado.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)

    sample = min(20_000, len(X_2d))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(X_2d), sample, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, labels, title in zip(axes,
                                  [iso_labels[idx], lof_labels[idx]],
                                  ["Isolation Forest", "Local Outlier Factor"]):
        normal  = labels == 1
        anomaly = labels == -1
        ax.scatter(X_2d[idx][normal,  0], X_2d[idx][normal,  1],
                   s=4, alpha=0.3, color="steelblue", label="Normal")
        ax.scatter(X_2d[idx][anomaly, 0], X_2d[idx][anomaly, 1],
                   s=10, alpha=0.7, color="red", label="Anomalia")
        ax.set_title(f"Anomalias — {title}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(markerscale=3)

    fig.suptitle("Detecção de Anomalias em Voos", fontsize=13, fontweight="bold")
    fig.savefig(ANOM_DIR / "anomaly_detection.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Figura de anomalias salva.")


def analyze_anomalous_flights(df: pd.DataFrame,
                               orig_idx: pd.Index,
                               iso_labels: np.ndarray) -> pd.DataFrame:
    """
    Caracteriza os voos anômalos: companhias e rotas mais presentes,
    médias de atraso.
    """
    df_sub = df.loc[orig_idx].copy()
    df_sub["IS_ANOMALY"] = (iso_labels == -1).astype(int)

    anomalies = df_sub[df_sub["IS_ANOMALY"] == 1]
    log.info(f"\nTop companhias com anomalias:")
    if "AIRLINE" in anomalies.columns:
        top_airlines = anomalies["AIRLINE"].value_counts().head(10)
        log.info(top_airlines.to_string())

    log.info(f"\nMédia de atraso — Anomalias vs Normal:")
    for col in ["ARRIVAL_DELAY", "DEPARTURE_DELAY"]:
        if col in df_sub.columns:
            means = df_sub.groupby("IS_ANOMALY")[col].mean()
            log.info(f"  {col}: Normal={means.get(0, float('nan')):.1f} | Anomalia={means.get(1, float('nan')):.1f}")

    df_sub.to_csv(ANOM_DIR / "flights_with_anomaly_flag.csv", index=False)
    return df_sub


def run_anomaly_detection(df: pd.DataFrame):
    """Pipeline completo de detecção de anomalias."""
    log.info("\n=== DETECÇÃO DE ANOMALIAS ===")
    X_scaled, orig_idx, feat_names = prepare_anomaly_data(df)

    iso_labels = run_isolation_forest(X_scaled)
    lof_labels = run_lof(X_scaled)

    plot_anomalies(X_scaled, iso_labels, lof_labels)
    df_result = analyze_anomalous_flights(df, orig_idx, iso_labels)

    log.info(f"Resultados de anomalia em {ANOM_DIR}")
    return df_result


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    df = preprocess_pipeline()
    run_anomaly_detection(df)
