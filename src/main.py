"""
main.py
=======
Orquestrador principal do pipeline de Machine Learning.

Uso:
    python main.py                    # executa pipeline completo
    python main.py --eda-only         # apenas EDA
    python main.py --supervised-only  # apenas modelos supervisionados
    python main.py --sample 0.02      # usa 2% dos dados (teste rápido)

Decisão de design:
  - Pipeline modular: cada etapa pode ser executada de forma independente.
  - Configuração centralizada no config.py.
  - Logging estruturado em arquivo + console.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Configura sys.path para imports relativos ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, MODEL_DIR, SAMPLE_FRAC
from data_preprocessing import preprocess_pipeline
from eda import run_full_eda
from supervised_models import run_supervised
from unsupervised_models import run_unsupervised
from anomaly_detection import run_anomaly_detection
from evaluation import run_evaluation

# ── Setup de logging ──────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

log_path = OUTPUT_DIR / "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Flight Delay ML Pipeline")
    parser.add_argument("--eda-only",         action="store_true", help="Executa apenas EDA")
    parser.add_argument("--supervised-only",  action="store_true", help="Executa apenas modelos supervisionados")
    parser.add_argument("--unsupervised-only",action="store_true", help="Executa apenas modelos não supervisionados")
    parser.add_argument("--anomaly-only",     action="store_true", help="Executa apenas detecção de anomalias")
    parser.add_argument("--sample",           type=float, default=None,
                        help="Fração de amostra (ex: 0.01 para 1%%)")
    return parser.parse_args()


def run_pipeline(args=None):
    t0 = time.time()
    log.info("=" * 60)
    log.info("INICIANDO PIPELINE — FLIGHT DELAY ML")
    log.info("=" * 60)

    sample_frac = args.sample if (args and args.sample) else SAMPLE_FRAC

    # ── 1. Pré-processamento ───────────────────────────────────────────────────
    log.info("\n[1/5] Pré-processamento de dados")
    df = preprocess_pipeline(sample_frac=sample_frac)

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    if not args or not (args.supervised_only or args.unsupervised_only or args.anomaly_only):
        log.info("\n[2/5] Análise Exploratória de Dados (EDA)")
        run_full_eda(df)

    if args and args.eda_only:
        log.info("Modo EDA-only. Encerrando.")
        return

    clf_results = reg_results = None

    # ── 3. Modelagem supervisionada ───────────────────────────────────────────
    if not args or not (args.unsupervised_only or args.anomaly_only):
        log.info("\n[3/5] Modelagem Supervisionada")
        clf_results, reg_results = run_supervised(df)

    if args and args.supervised_only:
        run_evaluation(clf_results, reg_results)
        log.info("Modo supervised-only. Encerrando.")
        return

    # ── 4. Modelagem não supervisionada ───────────────────────────────────────
    if not args or not (args.supervised_only or args.anomaly_only):
        log.info("\n[4/5] Modelagem Não Supervisionada (Clustering + PCA)")
        run_unsupervised(df)

    if args and args.unsupervised_only:
        log.info("Modo unsupervised-only. Encerrando.")
        return

    # ── 5. Detecção de anomalias ──────────────────────────────────────────────
    if not args or not args.supervised_only:
        log.info("\n[5/5] Detecção de Anomalias")
        run_anomaly_detection(df)

    # ── Relatório final ────────────────────────────────────────────────────────
    if clf_results is not None or reg_results is not None:
        run_evaluation(clf_results, reg_results)

    elapsed = time.time() - t0
    log.info(f"\nPipeline concluído em {elapsed/60:.1f} minutos.")
    log.info(f"Outputs em: {OUTPUT_DIR}")
    log.info(f"Modelos em: {MODEL_DIR}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
