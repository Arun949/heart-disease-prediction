# Quick Start Guide 🚀

## Setup (5 minutes)

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Add your data** to the `data/` folder:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`

## Run Training (One Command!)

```bash
python src/train.py
```

That's it! The script will:
- ✅ Load and preprocess data
- ✅ Engineer features
- ✅ Train 4 models with 10-fold CV
- ✅ Create ensemble predictions
- ✅ Generate submission file

## Output Files

After training, check the `outputs/` folder:
- `submission.csv` - Your predictions for Kaggle
- `oof_predictions.csv` - Out-of-fold predictions for analysis

## Expected Performance

**ROC-AUC Score**: ~0.89-0.91 (depending on data)

## Next Steps

### Want to explore data first?
```bash
python src/train.py --eda
```

### Want to tune hyperparameters?
```python
from src.hyperparameter_tuning import tune_hyperparameters
from src.data_preprocessing import get_data

X, y, X_test, _ = get_data()
best_params = tune_hyperparameters(X, y, n_trials=50)
```

### Want to use Jupyter?
```bash
jupyter notebook notebooks/exploration.ipynb
```

## Project Highlights

🎯 **4 Advanced Models**:
- XGBoost (Gradient Boosting)
- CatBoost (Categorical Boosting)
- LightGBM (Fast Boosting)
- Logistic Regression (Baseline)

🔥 **Smart Features**:
- Age groups
- Cholesterol risk categories
- Blood pressure categories
- Heart rate percentage
- Interaction features

💪 **Robust Validation**:
- 10-fold stratified cross-validation
- Out-of-fold predictions
- Weighted ensemble

## Troubleshooting

### Import Error?
Make sure you're in the project root directory:
```bash
cd heart-disease-prediction
python src/train.py
```

### Missing Data?
Place CSV files in the `data/` folder.

### Low Performance?
Try hyperparameter tuning or adjust weights in `config/config.py`.

## Questions?

Check the full `README.md` for detailed documentation.

---

**Happy Coding! 💻🫀**
