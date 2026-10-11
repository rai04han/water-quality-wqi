# Water Quality Classification (CCME WQI) — Group 7, 23MCAJ301

Random Forest classifier that predicts the CCME Water Quality Index class
(Excellent / Good / Fair / Marginal / Poor) from 8 measurements, with a Streamlit demo app.
Data: Hou Bay, China subset (2001–2017, 45,997 rows).

## Files
| File | Purpose |
|---|---|
| `Water_quality_complete_fixed.ipynb` | Full analysis and model comparison |
| `app.py` | Streamlit demo (Tab 1: predict, Tab 2: test on unseen 2013–2017 data) |
| `make_demo_models.py` | Builds the `.pkl` model files (run once) |
| `China_dataset.csv` | China subset of the dataset below |
| `requirements.txt` | Python packages |

## Run it (Python 3.10+)
Open a terminal **inside this folder**, then:

```
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python make_demo_models.py        # run once, creates the .pkl files
streamlit run app.py              # opens http://localhost:8501
```

## Data source and credit (required by the dataset licence)
The data in `China_dataset.csv` is a subset of:

- **Paper:** Karim et al. (2025). *A Comprehensive Dataset of Surface Water Quality Spanning 1940-2023 for Empirical and ML Adopted Research.* Scientific Data. https://doi.org/10.1038/s41597-025-04715-4
- **Dataset (Figshare):** Karim et al. (2025). *A Comprehensive Surface Water Quality Monitoring Dataset (1940-2023): 2.82 Million Record Resource for Empirical and ML-Based Research.* Figshare. https://doi.org/10.6084/m9.figshare.27800394

All credit for the data belongs to the original authors. This repository is a student project
and is not affiliated with them.
