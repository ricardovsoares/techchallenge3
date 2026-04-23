"""
eda.py
======
Análise Exploratória de Dados (EDA) — estatísticas descritivas e visualizações.

Decisões de design:
- Cada função gera uma figura independente e a salva em OUTPUT_DIR/eda/.
- Usamos seaborn + matplotlib para consistência visual.
- As visualizações foram escolhidas para responder às perguntas-guia do projeto:
    * Quais aeroportos são mais críticos?
    * Atrasos variam por dia da semana / horário?
    * Qual a distribuição e skewness de ARRIVAL_DELAY?
    * Quais causas de atraso são mais relevantes?
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sem display para ambientes headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

try:
    from config import OUTPUT_DIR
except ImportError:
    from src.config import OUTPUT_DIR

EDA_DIR = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = "viridis"
FIG_DPI  = 120


def _save(fig, name: str):
    path = EDA_DIR / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figura salva: {path}")


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas descritivas das principais colunas numéricas."""
    cols = ["ARRIVAL_DELAY", "DEPARTURE_DELAY", "DISTANCE",
            "SCHEDULED_TIME", "TAXI_OUT", "TAXI_IN"]
    stats = df[cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    log.info("\n" + stats.to_string())
    return stats


def plot_delay_distribution(df: pd.DataFrame):
    """
    Histograma + KDE de ARRIVAL_DELAY filtrado entre -60 e 300 min
    para melhor visualização (remover cauda extrema).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    data = df["ARRIVAL_DELAY"].clip(-60, 300)

    axes[0].hist(data, bins=80, color="#4C72B0", edgecolor="none", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", label="Sem atraso")
    axes[0].axvline(15, color="orange", linestyle="--", label="15 min (threshold)")
    axes[0].set_title("Distribuição do Atraso na Chegada")
    axes[0].set_xlabel("Atraso (min)")
    axes[0].set_ylabel("Frequência")
    axes[0].legend()

    # Proporção atrasados vs. não atrasados
    counts = df["DELAYED"].value_counts()
    axes[1].pie(counts, labels=["Não Atrasado", "Atrasado"], autopct="%1.1f%%",
                colors=["#4C72B0", "#DD8452"], startangle=90)
    axes[1].set_title("Proporção de Voos Atrasados (≥15 min)")

    fig.suptitle("Distribuição de Atrasos", fontsize=14, fontweight="bold")
    _save(fig, "01_delay_distribution")


def plot_delay_by_airline(df: pd.DataFrame, top_n: int = 15):
    """
    Boxplot mediano de atraso por companhia aérea (top N mais frequentes).
    Decisão: usar mediana em vez de média pois ARRIVAL_DELAY é muito assimétrico.
    """
    top_airlines = df["AIRLINE"].value_counts().nlargest(top_n).index
    subset = df[df["AIRLINE"].isin(top_airlines)]

    order = (subset.groupby("AIRLINE")["ARRIVAL_DELAY"]
                   .median().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=subset, x="AIRLINE", y="ARRIVAL_DELAY",
                order=order, palette=PALETTE, showfliers=False, ax=ax)
    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_title(f"Atraso na Chegada por Companhia Aérea (top {top_n})")
    ax.set_xlabel("Companhia Aérea")
    ax.set_ylabel("Atraso (min)")
    ax.set_ylim(-30, 60)
    _save(fig, "02_delay_by_airline")


def plot_delay_by_day_and_hour(df: pd.DataFrame):
    """
    Heatmap de atraso médio cruzando dia da semana × período do dia.
    Revela padrões temporais (ex.: sextas à noite têm mais atraso).
    """
    day_names = {1: "Seg", 2: "Ter", 3: "Qua", 4: "Qui",
                 5: "Sex", 6: "Sáb", 7: "Dom"}
    df2 = df.copy()
    df2["DAY_NAME"] = df2["DAY_OF_WEEK"].map(day_names)

    pivot = (df2.groupby(["DAY_NAME", "PERIOD_OF_DAY"])["ARRIVAL_DELAY"]
                .mean().unstack("PERIOD_OF_DAY"))

    day_order   = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    period_order = ["MADRUGADA", "MANHA", "TARDE", "NOITE"]
    pivot = pivot.reindex(index=day_order,
                          columns=[c for c in period_order if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Atraso médio (min)"})
    ax.set_title("Atraso Médio: Dia da Semana × Período do Dia")
    ax.set_xlabel("Período do Dia")
    ax.set_ylabel("Dia da Semana")
    _save(fig, "03_heatmap_day_period")


def plot_top_airports(df: pd.DataFrame, top_n: int = 20):
    """
    Aeroportos com maior atraso médio (mín. 500 voos para robustez estatística).
    """
    airport_stats = (df.groupby("ORIGIN_AIRPORT")
                       .agg(mean_delay=("ARRIVAL_DELAY", "mean"),
                            n_flights=("ARRIVAL_DELAY", "count"))
                       .query("n_flights >= 500")
                       .nlargest(top_n, "mean_delay")
                       .reset_index())

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(airport_stats["ORIGIN_AIRPORT"],
                   airport_stats["mean_delay"],
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, top_n)))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Top {top_n} Aeroportos com Maior Atraso Médio (≥500 voos)")
    ax.set_xlabel("Atraso médio (min)")
    ax.set_ylabel("Aeroporto de Origem")
    _save(fig, "04_top_airports_delay")


def plot_delay_causes(df: pd.DataFrame):
    """
    Barplot das causas de atraso (AIR_SYSTEM, SECURITY, AIRLINE,
    LATE_AIRCRAFT, WEATHER) mostrando contribuição média global.
    """
    cause_cols = {
        "AIR_SYSTEM_DELAY":   "Controle Aéreo",
        "SECURITY_DELAY":     "Segurança",
        "AIRLINE_DELAY":      "Companhia",
        "LATE_AIRCRAFT_DELAY":"Aeronave Atrasada",
        "WEATHER_DELAY":      "Clima",
    }
    cause_means = df[list(cause_cols.keys())].mean().rename(cause_cols)
    cause_means = cause_means.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(cause_means)))
    ax.barh(cause_means.index, cause_means.values, color=colors)
    ax.set_title("Contribuição Média por Causa de Atraso (todos os voos)")
    ax.set_xlabel("Atraso médio (min)")
    _save(fig, "05_delay_causes")


def plot_seasonal_trends(df: pd.DataFrame):
    """
    Atraso médio por mês — revela sazonalidade (pico em junho-julho
    e dezembro nos EUA por conta de férias e neve).
    """
    monthly = (df.groupby("MONTH")["ARRIVAL_DELAY"]
                 .agg(["mean", "median", "count"])
                 .reset_index())
    monthly.columns = ["MONTH", "mean_delay", "median_delay", "n_flights"]

    month_names = ["Jan","Fev","Mar","Abr","Mai","Jun",
                   "Jul","Ago","Set","Out","Nov","Dez"]
    monthly["MONTH_NAME"] = monthly["MONTH"].apply(lambda m: month_names[m-1])

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.bar(monthly["MONTH_NAME"], monthly["mean_delay"],
            color="#4C72B0", alpha=0.7, label="Atraso médio")
    ax1.plot(monthly["MONTH_NAME"], monthly["median_delay"],
             color="orange", marker="o", linewidth=2, label="Mediana")
    ax2.plot(monthly["MONTH_NAME"], monthly["n_flights"],
             color="green", linestyle="--", marker="s", linewidth=1.5, label="Nº voos")

    ax1.set_xlabel("Mês")
    ax1.set_ylabel("Atraso (min)")
    ax2.set_ylabel("Nº de Voos")
    ax1.set_title("Tendência Sazonal de Atrasos por Mês")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    _save(fig, "06_seasonal_trends")


def plot_correlation_matrix(df: pd.DataFrame):
    """
    Mapa de correlação entre variáveis numéricas relevantes.
    Ajuda a identificar multicolinearidade antes da modelagem.
    """
    num_cols = ["ARRIVAL_DELAY", "DEPARTURE_DELAY", "TAXI_OUT", "TAXI_IN",
                "SCHEDULED_TIME", "ELAPSED_TIME", "AIR_TIME", "DISTANCE",
                "AIR_SYSTEM_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY",
                "WEATHER_DELAY"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.3, ax=ax, annot_kws={"size": 7})
    ax.set_title("Matriz de Correlação — Variáveis Numéricas")
    _save(fig, "07_correlation_matrix")


def run_full_eda(df: pd.DataFrame):
    """Executa todas as análises EDA em sequência."""
    log.info("=== Iniciando EDA ===")
    descriptive_stats(df)
    plot_delay_distribution(df)
    plot_delay_by_airline(df)
    plot_delay_by_day_and_hour(df)
    plot_top_airports(df)
    plot_delay_causes(df)
    plot_seasonal_trends(df)
    plot_correlation_matrix(df)
    log.info(f"EDA concluída. Figuras em {EDA_DIR}")


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    df = preprocess_pipeline()
    run_full_eda(df)
