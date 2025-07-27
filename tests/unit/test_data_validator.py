# tests/unit/test_data_validator.py
"""🧪 Tests pour votre DataValidator générique"""

import pytest
import pandas as pd
import numpy as np
from src.data.data_validator import create_data_validator, ValidationResult

class TestDataValidator:
    """Tests pour le validator générique (pas spécialisé e-commerce)"""
    
    def test_validator_initialization(self):
        """🏗️ Test initialisation validator"""
        validator = create_data_validator()
        assert validator is not None
        assert hasattr(validator, 'required_columns')
    
    def test_valid_generic_dataset(self, generic_valid_data):
        """✅ Test dataset générique valide"""
        validator = create_data_validator()
        result = validator.validate_dataset(generic_valid_data, "test_dataset")
        
        assert isinstance(result, ValidationResult)
        assert result.quality_score >= 0
        assert result.quality_score <= 100
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.recommendations, list)
    
    def test_empty_dataset(self):
        """❌ Test dataset vide"""
        validator = create_data_validator()
        empty_df = pd.DataFrame()
        
        result = validator.validate_dataset(empty_df)
        assert not result.is_valid
        assert "Dataset vide" in str(result.errors) or len(result.errors) > 0
    
    def test_dataset_with_missing_values(self, data_with_missing):
        """⚠️ Test données avec valeurs manquantes"""
        validator = create_data_validator()
        result = validator.validate_dataset(data_with_missing)
        
        assert result.data_quality_metrics['missing_values_count'] > 0
        assert result.data_quality_metrics['missing_values_pct'] > 0
    
    def test_dataset_with_duplicates(self, data_with_duplicates):
        """🔄 Test données avec doublons"""
        validator = create_data_validator()
        result = validator.validate_dataset(data_with_duplicates)
        
        assert result.data_quality_metrics['duplicate_rows'] > 0
        assert result.data_quality_metrics['duplicate_rows_pct'] > 0
    
    def test_quality_score_calculation(self, generic_valid_data):
        """🏆 Test calcul score qualité"""
        validator = create_data_validator()
        result = validator.validate_dataset(generic_valid_data)
        
        assert 0 <= result.quality_score <= 100
        assert isinstance(result.quality_score, float)
    
    def test_quick_check_method(self, generic_valid_data):
        """⚡ Test méthode quick_check"""
        validator = create_data_validator()
        is_valid = validator.quick_check(generic_valid_data)
        
        assert isinstance(is_valid, bool)
    
    def test_validation_result_structure(self, generic_valid_data):
        """🏗️ Test structure ValidationResult"""
        validator = create_data_validator()
        result = validator.validate_dataset(generic_valid_data)
        
        # Vérifier tous les champs requis
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'quality_score')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'missing_columns')
        assert hasattr(result, 'data_quality_metrics')
        assert hasattr(result, 'recommendations')
    
    def test_data_quality_metrics_completeness(self, generic_valid_data):
        """📊 Test complétude des métriques"""
        validator = create_data_validator()
        result = validator.validate_dataset(generic_valid_data)
        
        metrics = result.data_quality_metrics
        
        # Métriques obligatoires
        expected_metrics = [
            'total_rows', 'total_columns', 'missing_values_count',
            'missing_values_pct', 'duplicate_rows', 'duplicate_rows_pct'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Métrique manquante: {metric}"
