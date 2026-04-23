"""
supervised_models.py
====================
Pipeline de modelagem supervisionada: classificação e regressão de atrasos.

Decisões de design:
───────────────────
Classificação (DELAYED 0/1):
  - Algoritmos comparados: RandomForestClassifier vs LightGBMClassifier
  - Métricas: F1-score (classe positiva), ROC-AUC, Precision, Recall,
    Confusion Matrix. Accuracy foi evitada por ser enganosa com classes
    desbalanceadas (~37% atrasados).
  - Balanceamento: class_weight="balanced" em ambos os modelos.
  - Pré-processamento: StandardScaler para numéricas + OrdinalEncoder
    para categóricas (LightGBM tolera OrdinalEncoder; RF também).

Regressão (ARRIVAL_DELAY em minutos):
  - Algoritmos: RandomForestRegressor vs LightGBMRegressor
  - Métricas: RMSE, MAE, R². RMSE penaliza erros grandes (importante
    do ponto de vista operacional).
  - Usamos apenas registros com ARRIVAL_DELAY > 0 para modelar
    a magnitude do atraso (filtro opcional configurável).

Feature Importance:
  - Extraída do modelo tree-based (impurity-based para RF,
    gain-based para LightGBM).
"""

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logging.warning("LightGBM não instalado. Usando apenas RandomForest.")

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

try:
    from config import (
        ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
        TARGET_CLASSIFICATION, TARGET_REGRESSION,
        RF_PARAMS, LGB_PARAMS, RF_REG_PARAMS, LGB_REG_PARAMS,
        TEST_SIZE, RANDOM_STATE, OUTPUT_DIR, MODEL_DIR,
    )
except ImportError:
    from src.config import (
        ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
        TARGET_CLASSIFICATION, TARGET_REGRESSION,
        RF_PARAMS, LGB_PARAMS, RF_REG_PARAMS, LGB_REG_PARAMS,
        TEST_SIZE, RANDOM_STATE, OUTPUT_DIR, MODEL_DIR,
    )

SUP_DIR = OUTPUT_DIR / "supervised"
SUP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Pré-processador ─────────────────────────────────────────────────────────
def build_preprocessor(cat_features, num_features):
    """
    ColumnTransformer com:
    - OrdinalEncoder para categóricas (compatível com tree-based models)
    - StandardScaler para numéricas (melhora convergência de alguns modelos)
    """
    categorical_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    numerical_transformer = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )


def prepare_data(df: pd.DataFrame, target: str, features=None):
    """Seleciona features/target, faz split estratificado (classificação) ou random (regressão)."""
    if features is None:
        features = ALL_FEATURES
    features = [f for f in features if f in df.columns]

    df_clean = df[features + [target]].dropna(subset=[target])
    X = df_clean[features]
    y = df_clean[target]

    stratify = y if target == TARGET_CLASSIFICATION else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )
    log.info(f"Treino: {X_train.shape} | Teste: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─── Classificação ────────────────────────────────────────────────────────────
def build_clf_pipeline(estimator):
    cat_feats = [f for f in ALL_FEATURES if f in CATEGORICAL_FEATURES]
    num_feats = [f for f in ALL_FEATURES if f in NUMERICAL_FEATURES]
    preprocessor = build_preprocessor(cat_feats, num_feats)
    return Pipeline([("preprocessor", preprocessor), ("clf", estimator)])


def evaluate_classifier(pipeline, X_test, y_test, name: str):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Não Atrasado", "Atrasado"])

    log.info(f"\n{'='*50}")
    log.info(f"Modelo: {name}")
    log.info(f"F1-Score (Atrasado): {f1:.4f}")
    log.info(f"ROC-AUC:             {auc:.4f}")
    log.info(f"\n{report}")

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Não Atrasado", "Atrasado"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.savefig(SUP_DIR / f"cm_{name.lower().replace(' ','_')}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {"name": name, "f1": f1, "auc": auc}


def plot_feature_importance(pipeline, feature_names: list, model_name: str, top_n: int = 20):
    """Extrai e plota feature importances do estimador dentro do pipeline."""
    estimator = pipeline.named_steps["clf"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    else:
        log.warning(f"Modelo {model_name} não tem feature_importances_.")
        return

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.nlargest(top_n, "importance")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi_df["feature"], fi_df["importance"], color="#4C72B0")
    ax.set_title(f"Top {top_n} Features — {model_name}")
    ax.set_xlabel("Importância")
    fig.savefig(SUP_DIR / f"fi_{model_name.lower().replace(' ','_')}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Feature importance salva para {model_name}.")


def run_classification(df: pd.DataFrame):
    """
    Treina RF e (se disponível) LightGBM para classificação de atraso.
    Retorna dict com métricas de ambos os modelos.
    """
    log.info("\n=== CLASSIFICAÇÃO ===")
    X_train, X_test, y_train, y_test = prepare_data(df, TARGET_CLASSIFICATION)

    # Nomes de features após o ColumnTransformer (num primeiro, depois cat)
    cat_feats = [f for f in ALL_FEATURES if f in CATEGORICAL_FEATURES and f in df.columns]
    num_feats = [f for f in ALL_FEATURES if f in NUMERICAL_FEATURES and f in df.columns]
    feature_names = num_feats + cat_feats

    results = []

    # Random Forest
    rf_clf = build_clf_pipeline(RandomForestClassifier(**RF_PARAMS))
    rf_clf.fit(X_train, y_train)
    res = evaluate_classifier(rf_clf, X_test, y_test, "Random Forest")
    plot_feature_importance(rf_clf, feature_names, "Random Forest")
    results.append(res)
    joblib.dump(rf_clf, MODEL_DIR / "rf_classifier.pkl")

    # LightGBM
    if HAS_LGB:
        lgb_clf = build_clf_pipeline(lgb.LGBMClassifier(**LGB_PARAMS))
        lgb_clf.fit(X_train, y_train)
        res = evaluate_classifier(lgb_clf, X_test, y_test, "LightGBM")
        plot_feature_importance(lgb_clf, feature_names, "LightGBM")
        results.append(res)
        joblib.dump(lgb_clf, MODEL_DIR / "lgb_classifier.pkl")

    # Comparativo
    results_df = pd.DataFrame(results)
    log.info(f"\nComparativo Classificação:\n{results_df.to_string(index=False)}")
    results_df.to_csv(SUP_DIR / "classification_results.csv", index=False)

    return results_df


# ─── Regressão ────────────────────────────────────────────────────────────────
def build_reg_pipeline(estimator):
    cat_feats = [f for f in ALL_FEATURES if f in CATEGORICAL_FEATURES]
    num_feats = [f for f in ALL_FEATURES if f in NUMERICAL_FEATURES]
    preprocessor = build_preprocessor(cat_feats, num_feats)
    return Pipeline([("preprocessor", preprocessor), ("reg", estimator)])


def evaluate_regressor(pipeline, X_test, y_test, name: str):
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    log.info(f"\n{'='*50}")
    log.info(f"Modelo: {name}")
    log.info(f"RMSE: {rmse:.2f} min")
    log.info(f"MAE:  {mae:.2f} min")
    log.info(f"R²:   {r2:.4f}")

    # Scatter plot real vs predito
    fig, ax = plt.subplots(figsize=(6, 6))
    sample_idx = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)
    y_test_arr = np.array(y_test)
    ax.scatter(y_test_arr[sample_idx], y_pred[sample_idx], alpha=0.3, s=8, color="#4C72B0")
    lim = [y_test_arr.min(), min(y_test_arr.max(), 300)]
    ax.plot(lim, lim, "r--", linewidth=1.5)
    ax.set_xlabel("Atraso Real (min)")
    ax.set_ylabel("Atraso Predito (min)")
    ax.set_title(f"Real vs Predito — {name}\nRMSE={rmse:.1f} | R²={r2:.3f}")
    ax.set_xlim(lim); ax.set_ylim(lim)
    fig.savefig(SUP_DIR / f"scatter_{name.lower().replace(' ','_')}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {"name": name, "rmse": rmse, "mae": mae, "r2": r2}


def run_regression(df: pd.DataFrame):
    """
    Treina RF e (se disponível) LightGBM para regressão do tempo de atraso.
    Usa apenas voos com ARRIVAL_DELAY > 0 para prever magnitude do atraso.
    """
    log.info("\n=== REGRESSÃO ===")
    df_reg = df[df[TARGET_REGRESSION] > 0].copy()
    log.info(f"Voos com atraso positivo: {len(df_reg):,}")

    X_train, X_test, y_train, y_test = prepare_data(df_reg, TARGET_REGRESSION)

    results = []

    rf_reg = build_reg_pipeline(RandomForestRegressor(**RF_REG_PARAMS))
    rf_reg.fit(X_train, y_train)
    res = evaluate_regressor(rf_reg, X_test, y_test, "Random Forest Reg")
    results.append(res)
    joblib.dump(rf_reg, MODEL_DIR / "rf_regressor.pkl")

    if HAS_LGB:
        lgb_reg = build_reg_pipeline(lgb.LGBMRegressor(**LGB_REG_PARAMS))
        lgb_reg.fit(X_train, y_train)
        res = evaluate_regressor(lgb_reg, X_test, y_test, "LightGBM Reg")
        results.append(res)
        joblib.dump(lgb_reg, MODEL_DIR / "lgb_regressor.pkl")

    results_df = pd.DataFrame(results)
    log.info(f"\nComparativo Regressão:\n{results_df.to_string(index=False)}")
    results_df.to_csv(SUP_DIR / "regression_results.csv", index=False)

    return results_df


def run_supervised(df: pd.DataFrame):
    clf_results = run_classification(df)
    reg_results = run_regression(df)
    return clf_results, reg_results


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    df = preprocess_pipeline()
    run_supervised(df)
