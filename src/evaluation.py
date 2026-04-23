"""
evaluation.py
=============
Módulo de avaliação consolidada e geração de relatório final.

Decisões de design:
───────────────────
- Consolida métricas de todos os modelos em um único relatório CSV/HTML.
- Gera visualização comparativa entre modelos (bar chart de métricas).
- Calcula curvas ROC para classificadores.
- Exporta sumário executivo em texto para facilitar apresentação.
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

try:
    from config import OUTPUT_DIR, MODEL_DIR
except ImportError:
    from src.config import OUTPUT_DIR, MODEL_DIR

EVAL_DIR = OUTPUT_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def compare_classifiers(results_df: pd.DataFrame):
    """
    Gera gráfico comparativo de F1 e ROC-AUC entre classificadores.
    """
    if results_df is None or results_df.empty:
        log.warning("Sem resultados de classificação para comparar.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, metric, title in zip(axes,
                                  ["f1", "auc"],
                                  ["F1-Score (classe Atrasado)", "ROC-AUC"]):
        if metric not in results_df.columns:
            continue
        bars = ax.bar(results_df["name"], results_df[metric],
                      color=colors[:len(results_df)], edgecolor="white", linewidth=0.5)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel(metric.upper())
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Comparativo de Classificadores", fontsize=13, fontweight="bold")
    fig.savefig(EVAL_DIR / "classifier_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Gráfico comparativo de classificadores salvo.")


def compare_regressors(results_df: pd.DataFrame):
    """Gráfico comparativo RMSE/MAE/R² entre regressores."""
    if results_df is None or results_df.empty:
        log.warning("Sem resultados de regressão para comparar.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    metrics = [("rmse", "RMSE (min)", False),
               ("mae",  "MAE (min)",  False),
               ("r2",   "R²",         True)]

    for ax, (metric, title, higher_better) in zip(axes, metrics):
        if metric not in results_df.columns:
            continue
        bars = ax.bar(results_df["name"], results_df[metric],
                      color=colors[:len(results_df)], edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel(title)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Comparativo de Regressores", fontsize=13, fontweight="bold")
    fig.savefig(EVAL_DIR / "regressor_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Gráfico comparativo de regressores salvo.")


def generate_text_report(clf_results: pd.DataFrame,
                          reg_results: pd.DataFrame) -> str:
    """
    Gera relatório textual com principais resultados e conclusões.
    """
    lines = [
        "=" * 60,
        "RELATÓRIO FINAL — TECH CHALLENGE FASE 3",
        "Flight Delays — Machine Learning Engineering",
        "=" * 60,
        "",
        "1. CLASSIFICAÇÃO (Prever se voo vai atrasar ≥15 min)",
        "-" * 50,
    ]

    if clf_results is not None and not clf_results.empty:
        best_clf = clf_results.loc[clf_results["auc"].idxmax()]
        lines += [
            f"  Melhor modelo: {best_clf['name']}",
            f"  ROC-AUC:  {best_clf['auc']:.4f}",
            f"  F1-Score: {best_clf['f1']:.4f}",
            "",
            "  Todos os modelos:",
        ]
        for _, row in clf_results.iterrows():
            lines.append(f"    {row['name']:25s} | AUC={row['auc']:.4f} | F1={row['f1']:.4f}")
    else:
        lines.append("  Resultados não disponíveis.")

    lines += [
        "",
        "2. REGRESSÃO (Prever tempo de atraso em minutos)",
        "-" * 50,
    ]

    if reg_results is not None and not reg_results.empty:
        best_reg = reg_results.loc[reg_results["rmse"].idxmin()]
        lines += [
            f"  Melhor modelo: {best_reg['name']}",
            f"  RMSE: {best_reg['rmse']:.2f} min",
            f"  MAE:  {best_reg['mae']:.2f} min",
            f"  R²:   {best_reg['r2']:.4f}",
            "",
            "  Todos os modelos:",
        ]
        for _, row in reg_results.iterrows():
            lines.append(f"    {row['name']:25s} | RMSE={row['rmse']:.2f} | MAE={row['mae']:.2f} | R²={row['r2']:.4f}")
    else:
        lines.append("  Resultados não disponíveis.")

    lines += [
        "",
        "3. PRINCIPAIS CONCLUSÕES",
        "-" * 50,
        "  • DEPARTURE_DELAY é o preditor mais forte de ARRIVAL_DELAY.",
        "  • Voos no período noturno e em dias de sexta-feira têm",
        "    maior probabilidade de atraso.",
        "  • Meses de junho, julho e dezembro apresentam picos de atraso.",
        "  • Atraso causado por aeronave atrasada (LATE_AIRCRAFT) é",
        "    a causa mais frequente — efeito cascata de voos anteriores.",
        "  • K-Means revelou clusters distintos de rotas curtas/rápidas",
        "    vs rotas longas/transcontinentais.",
        "",
        "4. LIMITAÇÕES",
        "-" * 50,
        "  • Dataset limitado ao ano de 2015 (sem generalização temporal).",
        "  • DEPARTURE_DELAY pode introduzir viés (dado disponível apenas",
        "    após o voo partir — atenção ao deployment).",
        "  • Dados meteorológicos detalhados não estão presentes.",
        "",
        "5. PRÓXIMOS PASSOS",
        "-" * 50,
        "  • Integrar dados de clima (NOAA) como feature externa.",
        "  • Explorar modelos de séries temporais (LSTM) para capturar",
        "    efeito cascata entre voos sequenciais.",
        "  • Construir pipeline de inferência em tempo real.",
        "  • Ajustar hiperparâmetros com Optuna/Bayesian Search.",
        "=" * 60,
    ]

    report = "\n".join(lines)
    report_path = EVAL_DIR / "final_report.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info(f"Relatório salvo em {report_path}")
    log.info("\n" + report)
    return report


def run_evaluation(clf_results=None, reg_results=None):
    """Executa avaliação consolidada e gera relatório."""
    log.info("\n=== AVALIAÇÃO FINAL ===")

    if clf_results is not None:
        compare_classifiers(clf_results)

    if reg_results is not None:
        compare_regressors(reg_results)

    report = generate_text_report(clf_results, reg_results)
    return report


if __name__ == "__main__":
    # Exemplo standalone: carrega CSVs de resultados se existirem
    clf_path = OUTPUT_DIR / "supervised" / "classification_results.csv"
    reg_path = OUTPUT_DIR / "supervised" / "regression_results.csv"

    clf_df = pd.read_csv(clf_path) if clf_path.exists() else None
    reg_df = pd.read_csv(reg_path) if reg_path.exists() else None

    run_evaluation(clf_df, reg_df)
