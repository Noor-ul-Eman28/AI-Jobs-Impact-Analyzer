# AI Jobs Salary Predictor — Project Files

## Files
- `AI_Jobs_Salary_ANN.ipynb` — full Jupyter notebook: cleaning, feature engineering, EDA, ANN training (regression + classification), Random Forest baselines, evaluation, and artifact export. **Already executed** — all outputs/plots are visible without re-running.
- `app.py` — Streamlit dashboard that loads the trained models and lets you predict salary interactively, explore the dataset, and compare model performance.
- `models/` — all saved artifacts the dashboard depends on (ANN models, Random Forest models, scalers, encoders, metadata, cleaned dataset).

## How to run the dashboard
1. Put `ai_jobs_dataset.csv` in the same folder as the notebook if you want to re-run it (not required to run the dashboard — the `models/` folder already has everything).
2. Install dependencies:
   ```
   pip install streamlit tensorflow scikit-learn pandas numpy matplotlib seaborn joblib
   ```
3. From this folder, run:
   ```
   streamlit run app.py
   ```
   (On Windows PowerShell, if `streamlit` isn't found on PATH: `python -m streamlit run app.py`)

## Results summary
- Random Forest R² ≈ 0.30 / MAE ≈ $42.6K | ANN R² ≈ 0.28 / MAE ≈ $43.2K (salary regression)
- Random Forest accuracy ≈ 41.4% | ANN accuracy ≈ 41.7% (salary band classification: Low/Mid/High/Executive, 4-class random baseline = 25%)
