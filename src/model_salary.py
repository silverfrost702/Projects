import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _safe_split_first(val: str) -> str:
    if not isinstance(val, str) or not val:
        return "Unknown"
    return val.split("|")[0].strip() or "Unknown"


def _get_top_skills(df: pd.DataFrame, top_n: int) -> List[str]:
    if "skills" not in df.columns:
        return []
    s = df["skills"].fillna("").str.split("|")
    counts: Dict[str, int] = {}
    for lst in s:
        for item in lst:
            item = item.strip()
            if not item:
                continue
            counts[item] = counts.get(item, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [k for k, _ in top]


def _build_feature_matrix(df: pd.DataFrame, top_skill_list: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    # Target
    y = pd.to_numeric(df["median_salary"], errors="coerce")

    # Base features
    X = pd.DataFrame(index=df.index)

    # Remote flag
    if "remote_allowed" in df.columns:
        X["remote_allowed"] = df["remote_allowed"].astype(bool).astype(int)
    else:
        X["remote_allowed"] = 0

    # Employee count
    if "employee_count" in df.columns:
        X["employee_count"] = pd.to_numeric(df["employee_count"], errors="coerce").fillna(0)
    else:
        X["employee_count"] = 0

    # Industry (first)
    if "industry_names" in df.columns:
        ind_first = df["industry_names"].apply(_safe_split_first)
        X = pd.concat([X, pd.get_dummies(ind_first, prefix="industry", dummy_na=False)], axis=1)

    # Top skills one-hot
    for skill in top_skill_list:
        X[f"skill_{skill}"] = df["skills"].fillna("").str.contains(fr"(^|\|){skill}(\||$)").astype(int) if "skills" in df.columns else 0

    # Fill potential NaNs
    X = X.fillna(0)

    # Align X and y
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def train_model(input_csv: str, out_dir: str, top_n_skills: int = 15, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
    _ensure_dir(out_dir)
    df = pd.read_csv(input_csv)

    if "median_salary" not in df.columns:
        raise ValueError("median_salary not found. Ensure you ran merge_data.py first.")

    top_skills = _get_top_skills(df, top_n_skills)
    X, y = _build_feature_matrix(df, top_skills)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    r2 = float(r2_score(y_test, pred))
    mae = float(mean_absolute_error(y_test, pred))

    # Save metrics and artifacts
    metrics_path = Path(out_dir) / "results.csv"
    pd.DataFrame([
        {"metric": "r2", "value": r2},
        {"metric": "mae", "value": mae},
    ]).to_csv(metrics_path, index=False)

    feat_imp = getattr(model, "feature_importances_", None)
    if feat_imp is not None:
        fi_df = pd.DataFrame({"feature": X.columns, "importance": feat_imp}).sort_values("importance", ascending=False)
        fi_df.to_csv(Path(out_dir) / "feature_importances.csv", index=False)

    joblib.dump(model, Path(out_dir) / "model.joblib")

    return {"r2": r2, "mae": mae}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RandomForest model to predict median salary")
    p.add_argument("--input", default="outputs/merged_jobs.csv", help="Path to merged dataset CSV")
    p.add_argument("--out-dir", default="outputs", help="Directory to write metrics and model")
    p.add_argument("--top-n-skills", type=int, default=15, help="Number of top skills to one-hot encode")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(args.input, args.out_dir, top_n_skills=args.top_n_skills)
    print(f"[model_salary] r2={metrics['r2']:.4f}  mae={metrics['mae']:.2f}")


if __name__ == "__main__":
    main()
