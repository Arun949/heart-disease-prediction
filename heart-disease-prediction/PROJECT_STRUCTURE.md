# Project Structure Overview

```
heart-disease-prediction/
│
├── 📁 .vscode/                          # VS Code configuration
│   ├── settings.json                    # Editor settings
│   └── launch.json                      # Debug configurations
│
├── 📁 config/                           # Configuration module
│   ├── __init__.py
│   └── config.py                        # All hyperparameters and settings
│
├── 📁 data/                             # Data directory
│   ├── .gitkeep
│   ├── train.csv                        # (Place your training data here)
│   ├── test.csv                         # (Place your test data here)
│   └── sample_submission.csv            # (Place your submission template here)
│
├── 📁 models/                           # Saved models directory
│   └── .gitkeep
│
├── 📁 notebooks/                        # Jupyter notebooks
│   └── exploration.ipynb                # Interactive exploration notebook
│
├── 📁 outputs/                          # Output files
│   ├── .gitkeep
│   ├── submission.csv                   # (Generated after training)
│   └── oof_predictions.csv              # (Generated after training)
│
├── 📁 src/                              # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py            # Data loading and preprocessing
│   ├── eda.py                           # Exploratory data analysis
│   ├── hyperparameter_tuning.py         # Optuna-based hyperparameter tuning
│   ├── model_training.py                # Model training and ensemble
│   ├── predict.py                       # Prediction on new data
│   └── train.py                         # Main training pipeline
│
├── 📁 tests/                            # Unit tests
│   ├── __init__.py
│   └── test_preprocessing.py            # Tests for preprocessing
│
├── 📄 .gitignore                        # Git ignore rules
├── 📄 README.md                         # Full documentation
├── 📄 QUICKSTART.md                     # Quick start guide
├── 📄 requirements.txt                  # Python dependencies
└── 📄 setup.py                          # Package setup file

```

## File Descriptions

### Configuration (`config/`)
- **config.py**: Central configuration file containing:
  - Data paths and file locations
  - Feature definitions (categorical, numerical, binary)
  - Model hyperparameters (tuned)
  - Ensemble weights
  - Random seeds and CV settings

### Source Code (`src/`)
- **data_preprocessing.py**: 
  - DataPreprocessor class
  - Feature engineering (age groups, cholesterol risk, etc.)
  - Data scaling and encoding
  - Train/test alignment

- **model_training.py**: 
  - ModelTrainer class
  - 10-fold stratified cross-validation
  - Ensemble training (XGBoost, CatBoost, LightGBM, Logistic)
  - OOF predictions and metrics

- **eda.py**: 
  - EDA class for exploratory analysis
  - Distribution plots
  - Correlation heatmaps
  - Outlier detection

- **hyperparameter_tuning.py**: 
  - HyperparameterTuner class
  - Optuna-based optimization
  - Objective functions for each model

- **train.py**: 
  - Main training pipeline
  - Command-line interface
  - End-to-end workflow

- **predict.py**: 
  - Prediction on new data
  - Model loading
  - Preprocessing pipeline

### Notebooks (`notebooks/`)
- **exploration.ipynb**: 
  - Interactive Jupyter notebook
  - Step-by-step exploration
  - Visualization and analysis

### Tests (`tests/`)
- **test_preprocessing.py**: 
  - Unit tests for preprocessing
  - Data validation tests

### Documentation
- **README.md**: Complete project documentation
- **QUICKSTART.md**: Quick start guide for beginners

### VS Code Configuration (`.vscode/`)
- **settings.json**: Python environment, linting, formatting
- **launch.json**: Debug configurations for training and prediction

## Key Features

✅ **Modular Design**: Clean separation of concerns
✅ **Type Hints**: Better code documentation
✅ **Error Handling**: Robust error management
✅ **Logging**: Detailed training progress
✅ **Testing**: Unit tests for critical components
✅ **Documentation**: Comprehensive docs
✅ **VS Code Integration**: Full IDE support
✅ **Git Ready**: Proper .gitignore configuration

## How Files Work Together

1. **config.py** → Provides settings to all modules
2. **data_preprocessing.py** → Loads and prepares data
3. **model_training.py** → Trains models using preprocessed data
4. **train.py** → Orchestrates the entire pipeline
5. **predict.py** → Uses trained models for new predictions
6. **exploration.ipynb** → Interactive interface to all modules

## Running the Project

### Simple Training
```bash
python src/train.py
```

### With EDA
```bash
python src/train.py --eda
```

### Make Predictions
```bash
python src/predict.py data/new_data.csv
```

### Run Tests
```bash
python -m pytest tests/
```

### Interactive Exploration
```bash
jupyter notebook notebooks/exploration.ipynb
```

---

**This structure ensures maintainability, scalability, and professional development practices! 🚀**
