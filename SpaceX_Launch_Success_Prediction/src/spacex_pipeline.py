#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


RANDOM_SEED = 42
sns.set(style="whitegrid", context="notebook")


ORBIT_COMPLEXITY_WEIGHTS = {
    # heuristic weights: higher is generally more energy/complex
    "VLEO": 0.8,
    "LEO": 1.0,
    "ISS": 1.2,
    "SSO": 1.4,
    "PO": 1.4,   # Polar
    "MEO": 1.8,
    "HEO": 2.0,
    "GTO": 2.2,
    "GEO": 2.4,
    "SO": 1.6,   # Solar orbit / special missions
    "ES-L1": 2.6,
}


@dataclass
class Config:
    input_csv: str
    out_dir: str
    model_type: str
    test_size: float = 0.2
    random_state: int = RANDOM_SEED


def ensure_dirs(base_out: str) -> Tuple[str, str]:
    plots_dir = os.path.join(base_out, "plots")
    models_dir = os.path.join(base_out, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    return plots_dir, models_dir


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse date features
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # Normalize some boolean/string flags to booleans
    def to_bool(x):
        if isinstance(x, str):
            if x.strip().lower() == "true":
                return True
            if x.strip().lower() == "false":
                return False
        return bool(x)

    for col in ["GridFins", "Reused", "Legs"]:
        df[col] = df[col].apply(to_bool).astype(int)

    # Clean categorical NA
    for col in ["Orbit", "LaunchSite", "LandingPad", "BoosterVersion"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": "Unknown", "": "Unknown", "None": "Unknown"})

    # PayloadMass numeric
    df["PayloadMass"] = pd.to_numeric(df["PayloadMass"], errors="coerce")

    # Block, Flights, ReusedCount numeric
    for col in ["Block", "Flights", "ReusedCount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Orbit complexity
    df["OrbitComplexity"] = df["Orbit"].map(ORBIT_COMPLEXITY_WEIGHTS).fillna(1.5)

    # Mission complexity score (simple, interpretable, scale-robust)
    # PayloadMass scaled in 10,000 kg units; Block in 0..5; Flights in 0..5 (cap)
    payload_component = (df["PayloadMass"].fillna(0.0) / 10000.0).clip(lower=0.0, upper=2.0)
    block_component = (df["Block"].fillna(0.0) / 5.0).clip(lower=0.0, upper=1.0)
    flights_component = (df["Flights"].fillna(0.0) / 5.0).clip(lower=0.0, upper=1.0)
    orbit_component = (df["OrbitComplexity"].fillna(1.5) / 3.0).clip(lower=0.0, upper=1.0)

    df["MissionComplexityScore"] = (
        0.45 * payload_component
        + 0.25 * orbit_component
        + 0.20 * block_component
        + 0.10 * flights_component
    )

    return df


def eda_plots(df: pd.DataFrame, plots_dir: str) -> None:
    # Success rate by Orbit
    plt.figure(figsize=(10, 5))
    success_by_orbit = df.groupby("Orbit")["Class"].mean().sort_values(ascending=False)
    sns.barplot(x=success_by_orbit.index, y=success_by_orbit.values, color="#4C78A8")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Success rate (Class=1)")
    plt.xlabel("Orbit")
    plt.title("Launch success rate by Orbit")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eda_success_by_orbit.png"), dpi=200)
    plt.close()

    # Success rate by LaunchSite
    plt.figure(figsize=(10, 5))
    success_by_site = df.groupby("LaunchSite")["Class"].mean().sort_values(ascending=False)
    sns.barplot(x=success_by_site.index, y=success_by_site.values, color="#72B7B2")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Success rate (Class=1)")
    plt.xlabel("Launch Site")
    plt.title("Launch success rate by Launch Site")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eda_success_by_site.png"), dpi=200)
    plt.close()

    # Payload mass by success (boxplot)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Class", y="PayloadMass", palette=["#E45756", "#4C78A8"])
    plt.xticks([0, 1], ["Failure", "Success"])
    plt.xlabel("Outcome")
    plt.ylabel("Payload mass (kg)")
    plt.title("Payload mass vs outcome")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eda_payload_vs_outcome.png"), dpi=200)
    plt.close()


def get_feature_sets() -> Tuple[List[str], List[str]]:
    numeric_features = [
        "FlightNumber",
        "PayloadMass",
        "Block",
        "ReusedCount",
        "Flights",
        "GridFins",
        "Reused",
        "Legs",
        "Year",
        "Month",
        "MissionComplexityScore",
    ]
    categorical_features = [
        "Orbit",
        "LaunchSite",
        "LandingPad",
        # BoosterVersion mostly constant here; Block covers evolution
    ]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model(model_type: str):
    model_type = model_type.lower()
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    if model_type == "xgb" and XGB_AVAILABLE:
        return XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    # Fallback
    return RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def get_ohe_feature_names(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    names = []
    # numeric names first (passthrough)
    names.extend(numeric_features)
    # categorical onehot names
    ohe = preprocessor.named_transformers_["cat"]
    ohe_names = list(ohe.get_feature_names_out(categorical_features))
    names.extend(ohe_names)
    return names


def train_and_evaluate(config: Config):
    plots_dir, models_dir = ensure_dirs(config.out_dir)

    df_raw = load_data(config.input_csv)
    df = preprocess(df_raw)

    # Drop rows missing target
    df = df.dropna(subset=["Class"]).copy()
    df["Class"] = df["Class"].astype(int)

    # Avoid leakage: drop text landing outcome
    if "Outcome" in df.columns:
        df = df.drop(columns=["Outcome"])  # contains True/False landing info

    numeric_features, categorical_features = get_feature_sets()

    X = df[numeric_features + categorical_features].copy()
    y = df["Class"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = build_model(config.model_type)

    pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model": config.model_type,
    }

    # Save metrics
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model pipeline
    joblib.dump(pipe, os.path.join(models_dir, f"model_{config.model_type}.joblib"))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # ROC curve
    try:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=200)
        plt.close()
    except Exception:
        pass

    # EDA plots on full dataset
    eda_plots(df, plots_dir)

    # Orbit vs predicted success probability (on test split)
    test_with_preds = X_test.copy()
    test_with_preds["pred_success_proba"] = y_proba
    test_with_preds["Orbit"] = X_test["Orbit"].values

    orbit_pred = test_with_preds.groupby("Orbit")["pred_success_proba"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=orbit_pred.index, y=orbit_pred.values, color="#F58518")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean predicted success probability")
    plt.xlabel("Orbit")
    plt.title("Predicted success probability by Orbit (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "orbit_vs_pred_success.png"), dpi=200)
    plt.close()

    # SHAP explanations (if available)
    if SHAP_AVAILABLE:
        try:
            # Extract fitted components
            fitted_pre: ColumnTransformer = pipe.named_steps["pre"]
            fitted_model = pipe.named_steps["model"]

            # Transform a sample of test data to speed up SHAP
            X_test_sample = X_test.sample(n=min(200, len(X_test)), random_state=config.random_state)
            X_test_trans = fitted_pre.transform(X_test_sample)

            feature_names = get_ohe_feature_names(fitted_pre, numeric_features, categorical_features)

            # Ensure dense array for SHAP
            X_array = np.asarray(X_test_trans)

            explainer = shap.Explainer(fitted_model, X_array)
            shap_values = explainer(X_array)

            # Summary dot plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values.values, X_array, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "shap_summary_dot.png"), dpi=200)
            plt.close()

            # Bar plot (mean absolute SHAP)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values.values, X_array, feature_names=feature_names, plot_type="bar", show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "shap_summary_bar.png"), dpi=200)
            plt.close()
        except Exception as e:
            # Log a note if SHAP fails, but don't crash the pipeline
            with open(os.path.join(models_dir, "shap_error.log"), "w") as f:
                f.write(str(e))

    # Save a small run manifest
    manifest = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_count_after_encoding": int(pipe.named_steps["pre"].transform(X.head(1)).shape[1]),
        "metrics": metrics,
    }
    with open(os.path.join(models_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(metrics, indent=2))


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="SpaceX Launch Success Prediction Pipeline")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "dataset_part_2.csv"),
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), ".."),
        help="Output directory (will create plots/ and models/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgb" if XGB_AVAILABLE else "rf",
        choices=["rf", "xgb"],
        help="Model type: rf or xgb",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    return Config(
        input_csv=os.path.abspath(args.input_csv),
        out_dir=os.path.abspath(args.out_dir),
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_and_evaluate(cfg)
