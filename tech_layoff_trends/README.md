# Tech Layoff Trend Analysis (2020–2025)

This project analyzes tech layoffs across companies, industries, and countries using the Kaggle Layoffs dataset (2020–2025). It includes:

- Monthly timeline of layoffs (Plotly)
- Which industries were hit hardest
- Correlation with funding stage and funds raised
- An interactive Top 10 Most Affected Companies dashboard

The analysis lives in the Jupyter notebook:
- `tech_layoff_trends/tech_layoffs_eda.ipynb`

## Quickstart

1) Create and activate a virtual environment (optional but recommended)
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

2) Install dependencies
```bash
pip install -U pip
pip install jupyter pandas numpy plotly ipywidgets nbformat openpyxl pyarrow kaleido
```

3) Launch Jupyter and open the notebook
```bash
jupyter notebook
```

4) Dataset location
- Place the dataset file inside `tech_layoff_trends/`.
- The notebook auto-discovers files with names containing "layoff" in the repo root and current folder.
- Alternatively set `DATASET_PATH` before launching or within the notebook.

## Exporting plots as images
The notebook renders Plotly charts interactively. To save images to `tech_layoff_trends/plots`, add this snippet after a figure is created:

```python
import pathlib
plots_dir = pathlib.Path('tech_layoff_trends/plots')
plots_dir.mkdir(parents=True, exist_ok=True)

fig_timeline.write_image(str(plots_dir / 'timeline.png'))
fig_industry.write_image(str(plots_dir / 'industry_impact.png'))
# For correlation heatmap and scatter (if created):
fig_corr.write_image(str(plots_dir / 'correlation_matrix.png'))
fig_scatter.write_image(str(plots_dir / 'layoffs_vs_funds.png'))
```

Note: image export requires `kaleido` (included above). If you see errors, run `pip install kaleido` in the active kernel and retry.

## Image gallery
Place generated images in `tech_layoff_trends/plots/` with these names (or update the markdown):

- Monthly Timeline
  
  ![Monthly Timeline](plots/timeline.png)

- Industry Impact (Top 20)
  
  ![Industry Impact](plots/industry_impact.png)

- Correlation Matrix
  
  ![Correlation Matrix](plots/correlation_matrix.png)

- Layoffs vs. Funds Raised
  
  ![Layoffs vs Funds](plots/layoffs_vs_funds.png)

## Insights (from the notebook)
If you've run the notebook cells, you can generate a reusable summary by running the provided "Insights Summary" cell in the notebook. It will produce an `INSIGHTS_SUMMARY.md` in this folder. You can paste key findings below:

- Peak month and year of layoffs
- Most affected industries and countries
- Top 10 companies by total layoffs
- Correlation between layoffs and funds raised / stages

> To update this section automatically, copy `INSIGHTS_SUMMARY.md` content here after running the generator cell in the notebook.
