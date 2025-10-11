import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[merge_data] Missing file: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def _first_present(columns: List[str], candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    return None


def _compute_median_salary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = df.columns.tolist()
    min_col = _first_present(cols, ["salary_min", "min_salary", "compensation_min", "min", "salary_minimum"])
    max_col = _first_present(cols, ["salary_max", "max_salary", "compensation_max", "max", "salary_maximum"])
    med_col = _first_present(cols, ["median_salary", "salary_median"])  # already provided

    if med_col is None:
        if min_col is not None and max_col is not None:
            df["median_salary"] = pd.to_numeric(df[min_col], errors="coerce").add(
                pd.to_numeric(df[max_col], errors="coerce"), fill_value=pd.NA
            ) / 2
        else:
            # best-effort fallback: try single 'salary' column
            one = _first_present(cols, ["salary", "compensation", "pay"])
            if one is not None:
                df["median_salary"] = pd.to_numeric(df[one], errors="coerce")
    else:
        df.rename(columns={med_col: "median_salary"}, inplace=True)

    return df


def build_skills_per_job(job_skills: pd.DataFrame, skills: pd.DataFrame) -> pd.DataFrame:
    if job_skills.empty or skills.empty:
        return pd.DataFrame(columns=["job_id", "skills"])

    # normalize column names
    js_job = _first_present(job_skills.columns.tolist(), ["job_id", "jobId", "job"])
    js_skill_id = _first_present(job_skills.columns.tolist(), ["skill_id", "skillId", "skill"])

    s_id = _first_present(skills.columns.tolist(), ["skill_id", "id", "skillId"]) \
        or _first_present(skills.columns.tolist(), ["id"])
    s_name = _first_present(skills.columns.tolist(), ["skill_name", "name", "skill"]) or "name"

    if js_job is None or js_skill_id is None or s_id is None or s_name is None:
        return pd.DataFrame(columns=["job_id", "skills"])

    merged = job_skills.merge(skills[[s_id, s_name]], left_on=js_skill_id, right_on=s_id, how="left")
    g = (
        merged.groupby(js_job)[s_name]
        .apply(lambda x: "|".join(sorted({str(v) for v in x.dropna()})))
        .reset_index()
        .rename(columns={js_job: "job_id", s_name: "skills"})
    )
    return g


def build_industries_per_job(job_industries: pd.DataFrame, industries: pd.DataFrame) -> pd.DataFrame:
    if job_industries.empty or industries.empty:
        return pd.DataFrame(columns=["job_id", "industry_names"])

    ji_job = _first_present(job_industries.columns.tolist(), ["job_id", "jobId", "job"]) \
        or _first_present(job_industries.columns.tolist(), ["job" ])
    ji_industry_id = _first_present(job_industries.columns.tolist(), ["industry_id", "industryId", "industry"]) \
        or _first_present(job_industries.columns.tolist(), ["industry"]) 

    i_id = _first_present(industries.columns.tolist(), ["industry_id", "id", "industryId"]) \
        or _first_present(industries.columns.tolist(), ["id"]) 
    i_name = _first_present(industries.columns.tolist(), ["industry_name", "name", "industry"]) or "name"

    if ji_job is None or ji_industry_id is None or i_id is None or i_name is None:
        return pd.DataFrame(columns=["job_id", "industry_names"])

    merged = job_industries.merge(industries[[i_id, i_name]], left_on=ji_industry_id, right_on=i_id, how="left")
    g = (
        merged.groupby(ji_job)[i_name]
        .apply(lambda x: "|".join(sorted({str(v) for v in x.dropna()})))
        .reset_index()
        .rename(columns={ji_job: "job_id", i_name: "industry_names"})
    )
    return g


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.lower()
    true_vals = {"true", "1", "yes", "y", "remote", "t"}
    return s.apply(lambda v: True if v in true_vals else False)


def merge_all(data_root: str) -> pd.DataFrame:
    root = Path(data_root)
    postings = _read_csv(root / "postings.csv")

    companies = _read_csv(root / "companies" / "companies.csv")
    employee_counts = _read_csv(root / "companies" / "employee_counts.csv")

    salaries = _read_csv(root / "jobs" / "salaries.csv")
    job_skills = _read_csv(root / "jobs" / "job_skills.csv")
    job_industries = _read_csv(root / "jobs" / "job_industries.csv")

    skills = _read_csv(root / "mappings" / "skills.csv")
    industries = _read_csv(root / "mappings" / "industries.csv")

    if postings.empty:
        raise FileNotFoundError(f"No postings.csv found in {root}")

    # Normalize required keys
    p_job = _first_present(postings.columns.tolist(), ["job_id", "id", "jobId"]) or "job_id"
    p_company = _first_present(postings.columns.tolist(), ["company_id", "companyId"]) or "company_id"

    # salaries
    salaries = _compute_median_salary(salaries.copy())
    s_job = _first_present(salaries.columns.tolist(), ["job_id", "jobId", "id"]) or "job_id"

    df = postings.merge(salaries, left_on=p_job, right_on=s_job, how="left")

    # companies
    c_id = _first_present(companies.columns.tolist(), ["company_id", "id", "companyId"]) or "company_id"
    df = df.merge(companies, left_on=p_company, right_on=c_id, how="left", suffixes=("", "_company"))

    # employee counts
    ec_company = _first_present(employee_counts.columns.tolist(), ["company_id", "companyId", "id"]) or "company_id"
    ec_count = _first_present(employee_counts.columns.tolist(), ["employee_count", "employees", "count"]) or "employee_count"
    if not employee_counts.empty:
        employee_counts = employee_counts[[ec_company, ec_count]].rename(
            columns={ec_company: "company_id", ec_count: "employee_count"}
        )
        df = df.merge(employee_counts, left_on=p_company, right_on="company_id", how="left")

    # skills and industries per job
    job_skill_text = build_skills_per_job(job_skills, skills)
    job_industry_text = build_industries_per_job(job_industries, industries)

    if not job_skill_text.empty:
        df = df.merge(job_skill_text, left_on=p_job, right_on="job_id", how="left", suffixes=("", "_skill"))
        if "job_id_skill" in df.columns:
            df.drop(columns=["job_id_skill"], inplace=True)

    if not job_industry_text.empty:
        df = df.merge(job_industry_text, left_on=p_job, right_on="job_id", how="left", suffixes=("", "_ind"))
        if "job_id_ind" in df.columns:
            df.drop(columns=["job_id_ind"], inplace=True)

    # normalize standard columns if present
    remote_col = _first_present(df.columns.tolist(), ["remote_allowed", "is_remote", "remote"]) or "remote_allowed"
    if remote_col not in df.columns:
        df[remote_col] = False
    df["remote_allowed"] = _coerce_bool(df[remote_col])

    if "employee_count" in df.columns:
        df["employee_count"] = pd.to_numeric(df["employee_count"], errors="coerce")

    # simple tidy columns
    # keep a focused set at the front for easier EDA
    preferred_front = [
        p_job,
        p_company,
        "median_salary",
        "remote_allowed",
        "employee_count",
        "skills",
        "industry_names",
    ]
    ordered = [c for c in preferred_front if c in df.columns] + [c for c in df.columns if c not in preferred_front]
    df = df[ordered]

    return df


def save_outputs(df: pd.DataFrame, output_csv: str, save_parquet: bool = True) -> Tuple[str, Optional[str]]:
    _ensure_dir(Path(output_csv).parent.as_posix())
    df.to_csv(output_csv, index=False)
    parquet_path = None
    if save_parquet:
        try:
            parquet_path = Path(output_csv).with_suffix(".parquet").as_posix()
            df.to_parquet(parquet_path, index=False)
        except Exception as exc:
            print(f"[merge_data] Skipped parquet export: {exc}")
    return output_csv, parquet_path


def maybe_download_drive_folder(gdrive_url: Optional[str], gdrive_folder_id: Optional[str], data_root: str) -> None:
    if not gdrive_url and not gdrive_folder_id:
        return
    try:
        import gdown  # type: ignore
    except Exception:
        print("[merge_data] gdown not installed. Install with: pip install gdown")
        return

    _ensure_dir(data_root)
    if gdrive_folder_id:
        url = f"https://drive.google.com/drive/folders/{gdrive_folder_id}"
    else:
        url = gdrive_url  # type: ignore

    print(f"[merge_data] Downloading Google Drive folder to {data_root} ...")
    try:
        gdown.download_folder(url=url, output=data_root, quiet=False, use_cookies=False)
    except Exception as exc:
        print(f"[merge_data] gdown.download_folder failed: {exc}")
        print("If the folder is not public, ensure permissions or download manually.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LinkedIn job market CSVs into a single dataset")
    p.add_argument("--data-root", default="data", help="Root directory containing CSV folders")
    p.add_argument("--gdrive-url", default=None, help="Optional Google Drive folder URL to download into data-root")
    p.add_argument("--gdrive-folder-id", default=None, help="Optional Google Drive folder ID to download into data-root")
    p.add_argument("--output", default="outputs/merged_jobs.csv", help="Output CSV path")
    p.add_argument("--no-parquet", action="store_true", help="Disable parquet export")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    maybe_download_drive_folder(args.gdrive_url, args.gdrive_folder_id, args.data_root)
    df = merge_all(args.data_root)
    csv_path, parquet_path = save_outputs(df, args.output, save_parquet=not args.no_parquet)
    print(f"[merge_data] Wrote: {csv_path}")
    if parquet_path:
        print(f"[merge_data] Wrote: {parquet_path}")


if __name__ == "__main__":
    main()
