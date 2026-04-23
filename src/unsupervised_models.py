"""
unsupervised_models.py
======================
Modelagem não supervisionada: clusterização (K-Means) + PCA.

Decisões de design:
───────────────────
Clusterização:
  - Algoritmo: K-Means (escalável a milhões de pontos via MiniBatchKMeans).
  - Features: variáveis do voo sem leakage (sem ARRIVAL_DELAY na hora de agrupar).
  - Determinação do K: Elbow Method (Inertia) + Silhouette Score.
  - Normalização com StandardScaler antes do K-Means
    (K-Means é sensível à escala das features).
  - Objetivo: agrupar ROTAS com perfis similares de distância, horário e duração.
    Isso permite identificar "clusters" de rotas problemáticas.

PCA:
  - Redução de dimensionalidade sobre as features numéricas.
  - Objetivo: visualizar a estrutura dos dados em 2D e 3D.
  - Mantemos componentes que explicam >= PCA_VARIANCE_THRESHOLD da variância.
  - Biplot dos dois primeiros componentes para interpretabilidade.
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

try:
    from config import (
        CLUSTER_FEATURES, N_CLUSTERS_RANGE, OPTIMAL_K,
        PCA_VARIANCE_THRESHOLD, RANDOM_STATE, OUTPUT_DIR,
    )
except ImportError:
    from src.config import (
        CLUSTER_FEATURES, N_CLUSTERS_RANGE, OPTIMAL_K,
        PCA_VARIANCE_THRESHOLD, RANDOM_STATE, OUTPUT_DIR,
    )

UNSUP_DIR = OUTPUT_DIR / "unsupervised"
UNSUP_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str):
    path = UNSUP_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura salva: {path}")


# ─── K-Means ──────────────────────────────────────────────────────────────────
def prepare_cluster_data(df: pd.DataFrame) -> tuple:
    """
    Seleciona features numéricas de clusterização, remove NaNs e normaliza.
    Retorna (X_scaled, scaler, feature_names).
    """
    feats = [f for f in CLUSTER_FEATURES if f in df.columns]
    X = df[feats].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log.info(f"Dados para clusterização: {X_scaled.shape}")
    return X_scaled, scaler, feats, X.index


def elbow_and_silhouette(X_scaled: np.ndarray,
                          k_range=None,
                          sample_size: int = 50_000) -> int:
    """
    Calcula Inertia e Silhouette para cada K em k_range.
    Usa amostra para eficiência.
    Retorna K sugerido pelo maior Silhouette Score.
    """
    if k_range is None:
        k_range = N_CLUSTERS_RANGE

    if len(X_scaled) > sample_size:
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[idx]
    else:
        X_sample = X_scaled

    inertias    = []
    silhouettes = []
    ks          = list(k_range)

    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=3, batch_size=10_000)
        labels = km.fit_predict(X_sample)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_sample, labels, sample_size=min(10_000, len(X_sample)),
                               random_state=RANDOM_STATE)
        silhouettes.append(sil)
        log.info(f"  K={k} | Inertia={km.inertia_:.0f} | Silhouette={sil:.4f}")

    # Plot Elbow + Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(ks, inertias, "bo-")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("Número de Clusters (K)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(ks, silhouettes, "rs-")
    best_k_idx = int(np.argmax(silhouettes))
    axes[1].axvline(ks[best_k_idx], color="green", linestyle="--",
                    label=f"Melhor K={ks[best_k_idx]}")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Número de Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    fig.suptitle("Determinação do Número Ideal de Clusters", fontsize=13, fontweight="bold")
    _save(fig, "01_elbow_silhouette")

    best_k = ks[best_k_idx]
    log.info(f"Melhor K pelo Silhouette: {best_k}")
    return best_k


def fit_kmeans(X_scaled: np.ndarray, k: int) -> MiniBatchKMeans:
    """Treina MiniBatchKMeans com o K definido."""
    km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                         n_init=10, batch_size=10_000, max_iter=300)
    km.fit(X_scaled)
    log.info(f"K-Means treinado com K={k}. Inertia final: {km.inertia_:.0f}")
    return km


def analyze_clusters(df: pd.DataFrame, labels: np.ndarray, feature_names: list):
    """
    Profila cada cluster: calcula médias das features + ARRIVAL_DELAY por cluster.
    Plota heatmap de perfil dos clusters.
    """
    df2 = df.copy()
    df2["CLUSTER"] = labels

    profile_cols = feature_names + (["ARRIVAL_DELAY"] if "ARRIVAL_DELAY" in df2.columns else [])
    profile = df2.groupby("CLUSTER")[profile_cols].mean()

    log.info(f"\nPerfil dos Clusters:\n{profile.to_string()}")
    profile.to_csv(UNSUP_DIR / "cluster_profiles.csv")

    # Heatmap normalizado
    from sklearn.preprocessing import MinMaxScaler
    profile_norm = pd.DataFrame(
        MinMaxScaler().fit_transform(profile),
        index=profile.index, columns=profile.columns
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    import seaborn as sns
    sns.heatmap(profile_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.4, ax=ax)
    ax.set_title("Perfil Normalizado dos Clusters (0=mín, 1=máx)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    _save(fig, "02_cluster_profiles_heatmap")

    # Distribuição de tamanho dos clusters
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(cluster_sizes.index, cluster_sizes.values,
            color=plt.cm.tab10(np.linspace(0, 1, len(cluster_sizes))))
    ax2.set_title("Tamanho dos Clusters")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Nº de voos")
    for i, v in enumerate(cluster_sizes.values):
        ax2.text(cluster_sizes.index[i], v + 50, f"{v:,}", ha="center", fontsize=9)
    _save(fig2, "03_cluster_sizes")

    return profile


def plot_clusters_pca2d(X_scaled: np.ndarray, labels: np.ndarray):
    """Reduz para 2D com PCA e plota os clusters."""
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca2.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=labels, cmap="tab10", alpha=0.3, s=4)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"Clusters visualizados em 2D (PCA)\nVariância explicada: {pca2.explained_variance_ratio_.sum():.1%}")
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
    _save(fig, "04_clusters_2d_pca")


# ─── PCA ──────────────────────────────────────────────────────────────────────
def run_pca(df: pd.DataFrame, variance_threshold: float = PCA_VARIANCE_THRESHOLD):
    """
    Aplica PCA sobre features numéricas.
    Plota: (1) variância explicada acumulada, (2) biplot PC1 vs PC2.
    Retorna o objeto PCA treinado e os componentes.
    """
    num_cols = ["ARRIVAL_DELAY", "DEPARTURE_DELAY", "TAXI_OUT", "TAXI_IN",
                "SCHEDULED_TIME", "DISTANCE", "AIR_TIME",
                "AIR_SYSTEM_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]
    num_cols = [c for c in num_cols if c in df.columns]

    X = df[num_cols].dropna()
    X_scaled = StandardScaler().fit_transform(X)

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)

    # Variância acumulada
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cum_var >= variance_threshold)) + 1
    log.info(f"PCA: {n_components} componentes explicam {cum_var[n_components-1]:.1%} da variância.")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(1, len(cum_var)+1), pca_full.explained_variance_ratio_,
           color="#4C72B0", alpha=0.7, label="Individual")
    ax.step(range(1, len(cum_var)+1), cum_var, where="mid",
            color="red", label="Acumulada")
    ax.axhline(variance_threshold, color="green", linestyle="--",
               label=f"{variance_threshold:.0%} threshold")
    ax.axvline(n_components, color="orange", linestyle=":",
               label=f"n={n_components} componentes")
    ax.set_title("Variância Explicada pelo PCA")
    ax.set_xlabel("Componente Principal")
    ax.set_ylabel("Proporção da Variância")
    ax.legend()
    _save(fig, "05_pca_variance")

    # Biplot PC1 × PC2
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca2.fit_transform(X_scaled)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sample_size = min(5000, len(X_2d))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(X_2d), sample_size, replace=False)
    ax2.scatter(X_2d[idx, 0], X_2d[idx, 1], alpha=0.2, s=5, color="steelblue")

    # Setas das features originais
    loadings = pca2.components_.T
    scale = 3.0
    for i, feat in enumerate(num_cols):
        ax2.annotate("", xy=(loadings[i, 0]*scale, loadings[i, 1]*scale),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
        ax2.text(loadings[i, 0]*scale*1.1, loadings[i, 1]*scale*1.1,
                 feat, fontsize=8, color="darkred")

    ax2.set_title(f"PCA Biplot — PC1 vs PC2\n({pca2.explained_variance_ratio_.sum():.1%} da variância)")
    ax2.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
    _save(fig2, "06_pca_biplot")

    return pca_full, n_components


def run_unsupervised(df: pd.DataFrame):
    """Pipeline completo: K-Means → análise de clusters → PCA."""
    log.info("\n=== MODELAGEM NÃO SUPERVISIONADA ===")

    # K-Means
    X_scaled, scaler, feat_names, orig_idx = prepare_cluster_data(df)
    best_k = elbow_and_silhouette(X_scaled)

    km = fit_kmeans(X_scaled, best_k)
    labels = km.labels_

    # Reconecta labels ao DataFrame original
    df_sub = df.loc[orig_idx].copy()
    analyze_clusters(df_sub, labels, feat_names)
    plot_clusters_pca2d(X_scaled, labels)

    # PCA
    run_pca(df)

    log.info(f"Análise não supervisionada concluída. Resultados em {UNSUP_DIR}")
    return km, labels


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    df = preprocess_pipeline()
    run_unsupervised(df)
