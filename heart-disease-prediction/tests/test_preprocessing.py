"""
Unit tests for data preprocessing
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Age': [50, 60, 45, 70],
            'Sex': [1, 0, 1, 0],
            'BP': [120, 140, 130, 150],
            'Cholesterol': [200, 240, 220, 260],
            'Max HR': [150, 140, 160, 130],
            'ST depression': [0.5, 1.0, 0.3, 1.5],
            'Chest pain type': ['Type 1', 'Type 2', 'Type 1', 'Type 3'],
            'Heart Disease': ['Absence', 'Presence', 'Absence', 'Presence']
        })
    
    def test_preprocess_target(self):
        """Test target variable preprocessing"""
        df = self.preprocessor.preprocess_target(self.sample_data.copy())
        
        # Check that target is converted to binary
        self.assertTrue(df['Heart Disease'].isin([0, 1]).all())
        self.assertEqual(df['Heart Disease'].iloc[0], 0)
        self.assertEqual(df['Heart Disease'].iloc[1], 1)
    
    def test_create_features(self):
        """Test feature engineering"""
        df = self.preprocessor.create_features(self.sample_data.copy())
        
        # Check that new features are created
        self.assertIn('Age_Group', df.columns)
        self.assertIn('Chol_Risk', df.columns)
        self.assertIn('BP_Category', df.columns)
        self.assertIn('HR_Percentage', df.columns)
        self.assertIn('Age_BP', df.columns)
    
    def test_scaler_fit(self):
        """Test that scaler is fitted correctly"""
        # This is a basic test - in production you'd want more comprehensive tests
        self.assertIsNotNone(self.preprocessor.scaler)


if __name__ == '__main__':
    unittest.main()
