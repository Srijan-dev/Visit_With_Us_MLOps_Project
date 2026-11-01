
"""Tune, evaluate, track and save the customer purchase prediction model."""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "ProdTaken"
FEATURES = ['Age', 'CityTier', 'Occupation', 'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome']
CATEGORICAL_FEATURES = ['Occupation', 'Gender', 'MaritalStatus', 'Designation']
NUMERIC_FEATURES = [col for col in FEATURES if col not in CATEGORICAL_FEATURES]

def train_and_evaluate():
    X_train = pd.read_csv(PROJECT_ROOT / "Xtrain.csv")
    X_test = pd.read_csv(PROJECT_ROOT / "Xtest.csv")
    y_train = pd.read_csv(PROJECT_ROOT / "ytrain.csv")[TARGET]
    y_test = pd.read_csv(PROJECT_ROOT / "ytest.csv")[TARGET]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=42, class_weight="balanced", n_jobs=1
        ))
    ])

    parameter_grid = {
        "classifier__n_estimators": [150],
        "classifier__max_depth": [None, 12],
        "classifier__min_samples_leaf": [1, 3]
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=3,
        n_jobs=1,
        refit=True,
        verbose=1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Keep the results for every tested parameter combination.
    grid_results = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    grid_results_path = PROJECT_ROOT / "deployment" / "grid_search_results.csv"
    grid_results_path.parent.mkdir(parents=True, exist_ok=True)
    grid_results.to_csv(grid_results_path, index=False)

    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities))
    }

    model_dir = PROJECT_ROOT / "deployment"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)

    results = {
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
        "test_metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist()
    }
    (model_dir / "training_results.json").write_text(
        json.dumps(results, indent=4), encoding="utf-8"
    )

    # MLflow is part of the supplied template. If it is not installed in a
    # local environment, keep a CSV experiment log so the notebook still runs.
    try:
        import mlflow
        mlflow.set_tracking_uri("file:" + str(PROJECT_ROOT / "mlruns"))
        mlflow.set_experiment("Visit_With_Us_Wellness_Tourism")
        with mlflow.start_run():
            mlflow.log_params(search.best_params_)
            mlflow.log_param("model", "RandomForestClassifier")
            mlflow.log_param("cv_strategy", "Stratified 3-fold")
            mlflow.log_param("parameter_grid", json.dumps(parameter_grid))
            mlflow.log_metrics(metrics)
            mlflow.log_metric("best_cv_f1", float(search.best_score_))
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(grid_results_path))
        print("Experiment details logged to MLflow.")
    except ImportError:
        log_row = {**search.best_params_, **metrics, "best_cv_f1": float(search.best_score_)}
        pd.DataFrame([log_row]).to_csv(
            PROJECT_ROOT / "deployment" / "experiment_log.csv", index=False
        )
        print("MLflow is not installed in this runtime; parameters, metrics and all grid-search results were saved locally.")

    print("\nBest parameters:", search.best_params_)
    print("Best cross-validation F1:", round(search.best_score_, 4))
    print("\nTest metrics:")
    print(pd.Series(metrics).round(4).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\\nSaved model:", model_path)
    return best_model, metrics, results

if __name__ == "__main__":
    train_and_evaluate()
