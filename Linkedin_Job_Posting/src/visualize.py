import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _explode_pipe(df: pd.DataFrame, col: str, new_col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[new_col])
    x = df[[col]].fillna("")
    x[new_col] = x[col].str.split("|")
    x = x.explode(new_col)
    x[new_col] = x[new_col].str.strip()
    x = x[x[new_col] != ""]
    return x[[new_col]]


def plot_highest_paying_skills(df: pd.DataFrame, out_dir: str, top_n: int = 20) -> None:
    if "skills" not in df.columns or "median_salary" not in df.columns:
        return
    x = df[["skills", "median_salary"]].copy()
    x["skill"] = x["skills"].fillna("").str.split("|")
    x = x.explode("skill")
    x["skill"] = x["skill"].str.strip()
    x = x[x["skill"] != ""]
    g = x.groupby("skill")["median_salary"].mean().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, max(4, top_n * 0.4)))
    sns.barplot(x=g.values, y=g.index, orient="h", palette="viridis")
    plt.xlabel("Average median salary")
    plt.ylabel("Skill")
    plt.title("Highest-paying skills (avg of median salary)")
    _ensure_dir(out_dir)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/bar_highest_paying_skills.png", dpi=150)
    plt.close()


def boxplot_salary_by_industry(df: pd.DataFrame, out_dir: str, top_n: int = 15) -> None:
    if "industry_names" not in df.columns or "median_salary" not in df.columns:
        return
    x = df[["industry_names", "median_salary"]].copy()
    x["industry"] = x["industry_names"].fillna("").str.split("|")
    x = x.explode("industry")
    x["industry"] = x["industry"].str.strip()
    x = x[x["industry"] != ""]
    top_inds = x["industry"].value_counts().head(top_n).index
    x = x[x["industry"].isin(top_inds)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=x, x="industry", y="median_salary")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Industry")
    plt.ylabel("Median salary")
    plt.title("Salary distribution by industry")
    _ensure_dir(out_dir)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/boxplot_salary_by_industry.png", dpi=150)
    plt.close()


def scatter_company_size_vs_salary(df: pd.DataFrame, out_dir: str) -> None:
    if "employee_count" not in df.columns or "median_salary" not in df.columns:
        return
    x = df[["employee_count", "median_salary"]].dropna().copy()
    x = x[(x["employee_count"] > 0) & (x["median_salary"] > 0)]
    if len(x) > 10000:
        x = x.sample(10000, random_state=0)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=x, x="employee_count", y="median_salary", alpha=0.5)
    plt.xscale("log")
    plt.xlabel("Employee count (log scale)")
    plt.ylabel("Median salary")
    plt.title("Company size vs median salary")
    _ensure_dir(out_dir)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/scatter_company_size_vs_salary.png", dpi=150)
    plt.close()


def pie_jobs_by_industry(df: pd.DataFrame, out_dir: str, top_n: int = 15) -> None:
    if "industry_names" not in df.columns:
        return
    x = _explode_pipe(df, "industry_names", "industry")
    if x.empty:
        return
    counts = x["industry"].value_counts()
    top = counts.head(top_n)
    other = counts.iloc[top_n:].sum()
    labels = list(top.index) + (["Other"] if other > 0 else [])
    sizes = list(top.values) + ([other] if other > 0 else [])

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Job postings by industry")
    _ensure_dir(out_dir)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pie_jobs_by_industry.png", dpi=150)
    plt.close()


def heatmap_salary_skill_x_industry(df: pd.DataFrame, out_dir: str, top_skills: int = 10, top_industries: int = 10) -> None:
    if "skills" not in df.columns or "industry_names" not in df.columns or "median_salary" not in df.columns:
        return
    s = df[["skills", "median_salary"]].copy()
    s["skill"] = s["skills"].fillna("").str.split("|")
    s = s.explode("skill")
    s["skill"] = s["skill"].str.strip()
    s = s[s["skill"] != ""]
    top_s = s["skill"].value_counts().head(top_skills).index

    i = df[["industry_names", "median_salary"]].copy()
    i["industry"] = i["industry_names"].fillna("").str.split("|")
    i = i.explode("industry")
    i["industry"] = i["industry"].str.strip()
    i = i[i["industry"] != ""]
    top_i = i["industry"].value_counts().head(top_industries).index

    m = df[["skills", "industry_names", "median_salary"]].copy()
    m["skill"] = m["skills"].fillna("").str.split("|")
    m = m.explode("skill")
    m["skill"] = m["skill"].str.strip()
    m = m[m["skill"] != ""]

    m["industry"] = m["industry_names"].fillna("").str.split("|")
    m = m.explode("industry")
    m["industry"] = m["industry"].str.strip()
    m = m[(m["skill"].isin(top_s)) & (m["industry"].isin(top_i))]

    pivot = m.pivot_table(index="skill", columns="industry", values="median_salary", aggfunc="mean")
    if pivot.empty:
        return

    plt.figure(figsize=(1.2 * len(pivot.columns) + 4, 0.5 * len(pivot.index) + 4))
    sns.heatmap(pivot, annot=False, cmap="mako", cbar_kws={"label": "Avg median salary"})
    plt.title("Average salary by skill × industry")
    plt.xlabel("Industry")
    plt.ylabel("Skill")
    _ensure_dir(out_dir)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/heatmap_salary_skill_x_industry.png", dpi=150)
    plt.close()


def generate_all(input_path: str, out_dir: str) -> None:
    _ensure_dir(out_dir)
    df = pd.read_csv(input_path)

    plot_highest_paying_skills(df, out_dir)
    boxplot_salary_by_industry(df, out_dir)
    scatter_company_size_vs_salary(df, out_dir)
    pie_jobs_by_industry(df, out_dir)
    heatmap_salary_skill_x_industry(df, out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate EDA charts from merged jobs dataset")
    p.add_argument("--input", default="outputs/merged_jobs.csv", help="Path to merged dataset CSV")
    p.add_argument("--out-dir", default="outputs/charts", help="Directory to write charts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_all(args.input, args.out_dir)
    print(f"[visualize] Charts saved to {args.out_dir}")


if __name__ == "__main__":
    main()
