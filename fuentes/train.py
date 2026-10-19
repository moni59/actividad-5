"""Entrena y compara Logistic Regression y Random Forest con MLflow."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from scipy.stats import ttest_rel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, ConfusionMatrixDisplay,
                             RocCurveDisplay, classification_report)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datos" / "datos_limp" / "breast_cancer_wisconsin_limpio.csv"
OUT = ROOT / "evidencias"
OUT.mkdir(exist_ok=True)
TARGET = "diagnostico"
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}
SEED = 42


def build_models(feature_names):
    num = list(feature_names)
    prep = ColumnTransformer([("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())]), num)], remainder="drop")
    return {
        "LogisticRegression": (Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=3000, random_state=SEED))]),
            {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]}),
        "RandomForest": (Pipeline([("prep", prep), ("model", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
            {"model__n_estimators": [200, 400], "model__max_depth": [None, 8],
             "model__min_samples_leaf": [1, 2], "model__class_weight": [None, "balanced"]})
    }


def main():
    if not DATA.exists():
        from datos_prep import exportar
        exportar()
    df = pd.read_csv(DATA)
    X, y = df.drop(columns=TARGET), df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("Actividad5_BreastCancer")
    summaries, fold_rows, fitted = [], [], {}
    for name, (pipe, grid) in build_models(X.columns).items():
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1,
                              refit=True, return_train_score=False)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        y_hat = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_hat),
            "precision": precision_score(y_test, y_hat, zero_division=0),
            "recall": recall_score(y_test, y_hat, zero_division=0),
            "f1": f1_score(y_test, y_hat, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "cv_roc_auc_mean": search.best_score_,
            "cv_roc_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        }
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"dataset": "Breast Cancer Wisconsin Diagnostic (scikit-learn)",
                             "positive_class": "benigno (1)", "split": "stratified 80/20", "cv": "5-fold stratified"})
            mlflow.log_params({"model": name, "seed": SEED, "test_size": 0.2,
                               "cv_folds": 5, "search_scoring": "roc_auc", **search.best_params_})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="pickle")
            mlflow.log_dict(classification_report(y_test, y_hat, target_names=["maligno", "benigno"], output_dict=True), "classification_report.json")
            ConfusionMatrixDisplay.from_predictions(y_test, y_hat, display_labels=["maligno", "benigno"], cmap="Blues")
            plt.title(f"Matriz de confusión — {name}"); plt.tight_layout()
            cm_file = OUT / f"confusion_{name}.png"; plt.savefig(cm_file, dpi=150); plt.close()
            mlflow.log_artifact(str(cm_file))
            fitted[name] = (model, metrics, run.info.run_id, y_prob)
        summaries.append({"modelo": name, **metrics, "mejores_parametros": json.dumps(search.best_params_, sort_keys=True)})
        cvres = pd.DataFrame(search.cv_results_)
        # Las 5 puntuaciones de CV de la mejor configuración se usan en el contraste pareado.
        idx = search.best_index_
        for j in range(5):
            fold_rows.append({"modelo": name, "fold": j+1,
                              "roc_auc": float(cvres.loc[idx, f"split{j}_test_score"])})
    result = pd.DataFrame(summaries)
    result.to_csv(OUT / "metricas_modelos.csv", index=False)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT / "roc_auc_por_fold.csv", index=False)
    names = list(fitted)
    a = [x["roc_auc"] for x in folds[folds.modelo == names[0]].sort_values("fold").to_dict("records")]
    b = [x["roc_auc"] for x in folds[folds.modelo == names[1]].sort_values("fold").to_dict("records")]
    test = ttest_rel(a, b)
    stats = {"prueba": "t pareada bilateral sobre ROC-AUC de 5 folds (exploratoria)",
             "modelo_1": names[0], "modelo_2": names[1], "t": float(test.statistic),
             "p_value": float(test.pvalue), "diferencia_media": float(np.mean(np.array(a)-np.array(b))),
             "nota": "Con solo cinco folds la inferencia es inestable; interpretar como comparación exploratoria."}
    (OUT / "comparacion_estadistica.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    plt.figure(figsize=(7, 5))
    for name, (model, _, _, prob) in fitted.items():
        RocCurveDisplay.from_predictions(y_test, prob, name=name, ax=plt.gca())
    plt.plot([0,1],[0,1],"k--",alpha=.5); plt.title("Curvas ROC en conjunto de prueba"); plt.tight_layout()
    roc_file = OUT / "curvas_roc.png"; plt.savefig(roc_file, dpi=160); plt.close()
    with mlflow.start_run(run_name="comparacion_final"):
        mlflow.log_artifact(str(OUT / "metricas_modelos.csv"))
        mlflow.log_artifact(str(OUT / "roc_auc_por_fold.csv"))
        mlflow.log_artifact(str(OUT / "comparacion_estadistica.json"))
        mlflow.log_artifact(str(roc_file))
        mlflow.log_metrics({"paired_t": stats["t"], "paired_p_value": stats["p_value"]})
    print(result.to_string(index=False))
    print(json.dumps(stats, indent=2))
    print(f"MLflow UI: mlflow ui --backend-store-uri {ROOT / 'mlruns'}")

if __name__ == "__main__":
    main()
