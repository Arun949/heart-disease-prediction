# Heart Disease Prediction Project 🫀

A comprehensive machine learning project for predicting heart disease probability using ensemble learning techniques.

## 🎯 Project Overview

This project implements a robust ensemble learning pipeline that combines multiple state-of-the-art machine learning models to predict heart disease probability. The solution includes:

- **Advanced Feature Engineering**: Creates interaction features and categorical transformations
- **Ensemble Learning**: Combines XGBoost, CatBoost, LightGBM, and Logistic Regression
- **Cross-Validation**: 10-fold stratified cross-validation for robust evaluation
- **Hyperparameter Optimization**: Optuna-based automatic hyperparameter tuning

## 📁 Project Structure

```
heart-disease-prediction/
│
├── config/
│   ├── __init__.py
│   └── config.py                 # Configuration and hyperparameters
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── model_training.py         # Model training and ensemble
│   ├── eda.py                    # Exploratory data analysis
│   ├── hyperparameter_tuning.py  # Optuna-based hyperparameter tuning
│   ├── train.py                  # Main training pipeline
│   └── predict.py                # Prediction on new data
│
├── data/
│   ├── train.csv                 # Training data
│   ├── test.csv                  # Test data
│   └── sample_submission.csv     # Submission template
│
├── notebooks/
│   └── exploration.ipynb         # Jupyter notebook for exploration
│
├── models/                       # Saved model files
├── outputs/                      # Results and submissions
├── tests/                        # Unit tests
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone or download the project**

2. **Install dependencies**:
```bash
cd heart-disease-prediction
pip install -r requirements.txt
```

3. **Place your data files** in the `data/` directory:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`

## 💻 Usage

### 1. Basic Training

Train the ensemble model with default settings:

```bash
python src/train.py
```

### 2. Training with EDA

Perform exploratory data analysis before training:

```bash
python src/train.py --eda
```

### 3. Custom Output Directory

Specify a custom output directory:

```bash
python src/train.py --output-dir ./my_results
```

### 4. Hyperparameter Tuning

For hyperparameter optimization (requires more time):

```python
from src.hyperparameter_tuning import tune_hyperparameters
from src.data_preprocessing import get_data

X, y, X_test, _ = get_data()
best_params = tune_hyperparameters(X, y, n_trials=100)
```

### 5. Making Predictions on New Data

```bash
python src/predict.py data/new_data.csv --output predictions.csv
```

## 🔧 Configuration

All model parameters and settings can be customized in `config/config.py`:

```python
# Adjust cross-validation folds
N_SPLITS = 10

# Modify ensemble weights
ENSEMBLE_WEIGHTS = {
    'xgboost': 0.35,
    'catboost': 0.35,
    'lightgbm': 0.20,
    'logistic': 0.10
}

# Update model hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    # ... more parameters
}
```

## 📊 Features

### Data Preprocessing
- Automatic handling of categorical variables
- RobustScaler for outlier-resistant normalization
- Feature engineering:
  - Age groups
  - Cholesterol risk categories
  - Blood pressure categories
  - HR percentage of age-predicted max
  - Interaction features (Age×BP, Age×Cholesterol, BP×Cholesterol)

### Models

1. **XGBoost**: Gradient boosting with extreme optimization
2. **CatBoost**: Gradient boosting optimized for categorical features
3. **LightGBM**: Fast gradient boosting framework
4. **Logistic Regression**: Linear baseline model

### Ensemble Strategy

- **Weighted Averaging**: Combines predictions from all models using optimized weights
- **Stratified K-Fold CV**: Ensures balanced class distribution across folds
- **Out-of-Fold Predictions**: Generates unbiased validation predictions

## 📈 Performance Metrics

The model is evaluated using:
- **ROC-AUC Score**: Primary metric for model comparison
- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall

## 🔍 Key Improvements Over Original Notebook

1. ✅ **Fixed Data Leakage**: Scaler fitted only on training data
2. ✅ **Modular Code Structure**: Clean, reusable components
3. ✅ **Enhanced Feature Engineering**: Additional meaningful features
4. ✅ **More Models**: Added LightGBM to ensemble
5. ✅ **Better Cross-Validation**: Increased to 10 folds
6. ✅ **Hyperparameter Optimization**: Optuna integration
7. ✅ **Comprehensive Logging**: Detailed training progress
8. ✅ **Configuration Management**: Centralized settings
9. ✅ **OOF Predictions**: Saved for model analysis
10. ✅ **Production-Ready**: Modular and deployable

## 📝 Example Output

```
================================================================================
TRAINING ENSEMBLE WITH 10-FOLD CROSS-VALIDATION
================================================================================

Fold 1/10
----------------------------------------
  xgboost      - ROC-AUC: 0.89234
  catboost     - ROC-AUC: 0.89456
  lightgbm     - ROC-AUC: 0.88923
  logistic     - ROC-AUC: 0.86234

  Ensemble     - ROC-AUC: 0.89678

...

================================================================================
CROSS-VALIDATION RESULTS
================================================================================

Individual Model OOF ROC-AUC Scores:
  xgboost     : 0.89123
  catboost    : 0.89345
  lightgbm    : 0.88834
  logistic    : 0.86123

Weighted Ensemble OOF ROC-AUC: 0.89567
================================================================================
```

## 🧪 Testing

Run unit tests:

```bash
python -m pytest tests/
```

## 📦 Dependencies

Key libraries:
- `scikit-learn`: Machine learning utilities
- `xgboost`: Gradient boosting
- `catboost`: Gradient boosting for categorical features
- `lightgbm`: Fast gradient boosting
- `optuna`: Hyperparameter optimization
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `matplotlib/seaborn`: Visualization

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is provided as-is for educational and research purposes.

## 👥 Author

Created for heart disease prediction using advanced machine learning techniques.

## 🙏 Acknowledgments

- Original Kaggle notebook inspiration
- Scikit-learn and ensemble learning community
- XGBoost, CatBoost, and LightGBM developers

---

**Happy Predicting! 🫀💻**
