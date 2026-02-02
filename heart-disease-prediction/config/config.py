"""
Configuration file for Heart Disease Prediction Project
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Data files
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

# Model configuration
RANDOM_STATE = 42
N_SPLITS = 10  # Increased from 5 for better validation
TEST_SIZE = 0.15

# Feature definitions
CAT_COLS = ['Chest pain type', 'EKG results', 'Slope of ST', 
            'Number of vessels fluro', 'Thallium']

NUM_COLS = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']

BINARY_COLS = ['Sex', 'FBS over 120', 'Exercise angina']

TARGET_COL = 'Heart Disease'
ID_COL = 'id'

# Model hyperparameters (tuned)
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
    'eval_metric': 'auc',
    'verbosity': 0
}

CATBOOST_PARAMS = {
    'iterations': 500,
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'random_state': RANDOM_STATE,
    'loss_function': 'Logloss',
    'verbose': 0,
    'early_stopping_rounds': 50
}

LIGHTGBM_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'verbose': -1
}

LOGISTIC_PARAMS = {
    'C': 0.1,
    'penalty': 'l2',
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': RANDOM_STATE
}

# Ensemble weights (can be tuned)
ENSEMBLE_WEIGHTS = {
    'xgboost': 0.35,
    'catboost': 0.35,
    'lightgbm': 0.20,
    'logistic': 0.10
}
