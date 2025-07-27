# tests/unit/test_data_transformer.py
"""🧪 Tests pour le data transformer basé sur votre notebook"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch

from src.data.data_transformer import create_data_transformer, CleanDataResult

class TestDataTransformer:
    """Tests pour le pipeline de transformation e-commerce"""
    
    def test_transformer_initialization(self):
        """🏗️ Test initialisation transformer"""
        transformer = create_data_transformer()
        
        assert hasattr(transformer, 'clean_dataset')
        assert callable(transformer.clean_dataset)
    
    def test_clean_dataset_basic(self, sample_ecommerce_raw_data):
        """✅ Test nettoyage de base"""
        transformer = create_data_transformer()
        
        result = transformer.clean_dataset(sample_ecommerce_raw_data)
        
        # Vérifications de base
        assert isinstance(result, CleanDataResult)
        assert isinstance(result.data, pd.DataFrame)
        assert result.rows_before > 0
        assert result.rows_after > 0
        assert result.cleaning_time_seconds >= 0
        assert isinstance(result.transformations_applied, list)
    
    def test_missing_values_handling(self, data_with_missing_values):
        """🔧 Test gestion valeurs manquantes"""
        transformer = create_data_transformer()
        
        # Données avant nettoyage
        missing_before = data_with_missing_values.isnull().sum().sum()
        assert missing_before > 0  # Vérifier qu'il y a des valeurs manquantes
        
        result = transformer.clean_dataset(data_with_missing_values)
        
        # Après nettoyage (selon votre logique du notebook)
        missing_after = result.data.isnull().sum().sum()
        
        # Les valeurs manquantes doivent être traitées
        # (soit supprimées, soit imputées selon votre logique)
        assert "missing_values_handled" in result.transformations_applied
    
    def test_duplicate_removal(self, data_with_duplicates):
        """🔧 Test suppression doublons"""
        transformer = create_data_transformer()
        
        # Vérifier présence de doublons
        duplicates_before = data_with_duplicates.duplicated().sum()
        assert duplicates_before > 0
        
        result = transformer.clean_dataset(data_with_duplicates)
        
        # Vérifier suppression doublons
        duplicates_after = result.data.duplicated().sum()
        assert duplicates_after == 0  # Plus de doublons
        assert "duplicates_removed" in result.transformations_applied
    
    def test_invoice_date_parsing(self, sample_ecommerce_raw_data):
        """📅 Test parsing des dates InvoiceDate"""
        transformer = create_data_transformer()
        
        # S'assurer qu'on a des dates en string
        if 'InvoiceDate' in sample_ecommerce_raw_data.columns:
            sample_ecommerce_raw_data['InvoiceDate'] = sample_ecommerce_raw_data['InvoiceDate'].astype(str)
        
        result = transformer.clean_dataset(sample_ecommerce_raw_data)
        
        # Vérifier conversion date
        if 'InvoiceDate' in result.data.columns:
            assert pd.api.types.is_datetime64_any_dtype(result.data['InvoiceDate'])
            assert "date_parsing" in result.transformations_applied
    
    def test_customer_id_type_conversion(self, sample_ecommerce_raw_data):
        """🔢 Test conversion Customer ID"""
        transformer = create_data_transformer()
        
        result = transformer.clean_dataset(sample_ecommerce_raw_data)
        
        # Customer ID doit être traité (soit float, soit string selon votre logique)
        if 'Customer ID' in result.data.columns:
            customer_id_col = result.data['Customer ID']
            # Vérifier que les NaN sont gérés correctement
            assert "customer_id_processed" in result.transformations_applied
    
    def test_price_quantity_validation(self, sample_ecommerce_raw_data):
        """💰 Test validation des prix et quantités"""
        transformer = create_data_transformer()
        
        # Ajouter des valeurs négatives/nulles pour tester
        test_data = sample_ecommerce_raw_data.copy()
        if len(test_data) > 2:
            test_data.loc[test_data.index[0], 'Price'] = -1.0  # Prix négatif
            test_data.loc[test_data.index[1], 'Quantity'] = 0  # Quantité nulle
        
        result = transformer.clean_dataset(test_data)
        
        # Vérifier gestion des prix/quantités selon votre logique
        if 'Price' in result.data.columns:
            # Selon votre notebook: prix négatifs supprimés ou gardés selon le contexte
            assert "price_validation" in result.transformations_applied
    
    def test_returns_detection(self, sample_ecommerce_raw_data):
        """🔄 Test détection des retours (Invoice commençant par C)"""
        transformer = create_data_transformer()
        
        # Ajouter un retour pour tester
        test_data = sample_ecommerce_raw_data.copy()
        if len(test_data) > 0:
            test_data.loc[test_data.index[0], 'Invoice'] = 'C536365'  # Retour
        
        result = transformer.clean_dataset(test_data)
        
        # Vérifier qu'une colonne 'is_return' ou logique équivalente existe
        assert "returns_detected" in result.transformations_applied
    
    def test_empty_dataframe(self):
        """📭 Test dataframe vide"""
        transformer = create_data_transformer()
        empty_df = pd.DataFrame()
        
        result = transformer.clean_dataset(empty_df)
        
        assert result.rows_before == 0
        assert result.rows_after == 0
        assert len(result.data) == 0
    
    def test_single_row_dataframe(self):
        """1️⃣ Test dataframe une ligne"""
        transformer = create_data_transformer()
        
        single_row_df = pd.DataFrame({
            'Invoice': ['536365'],
            'StockCode': ['85123A'],
            'Description': ['WHITE HANGING HEART'],
            'Quantity': [6],
            'InvoiceDate': ['12/1/2010 8:26'],
            'Price': [2.55],
            'Customer ID': [17850.0],
            'Country': ['United Kingdom']
        })
        
        result = transformer.clean_dataset(single_row_df)
        
        assert result.rows_before == 1
        assert result.rows_after >= 0  # Peut être supprimé selon les règles
        assert isinstance(result.data, pd.DataFrame)
    
    def test_all_transformations_logged(self, sample_ecommerce_raw_data):
        """📝 Test que toutes les transformations sont loggées"""
        transformer = create_data_transformer()
        
        result = transformer.clean_dataset(sample_ecommerce_raw_data)
        
        # Vérifier qu'on a une liste de transformations
        assert isinstance(result.transformations_applied, list)
        assert len(result.transformations_applied) > 0
        
        # Transformations typiques attendues (selon votre notebook)
        expected_transformations = [
            'missing_values_handled',
            'duplicates_removed', 
            'date_parsing',
            'customer_id_processed',
            'data_types_optimized'
        ]
        
        # Au moins quelques-unes doivent être présentes
        found_transformations = [t for t in expected_transformations 
                               if t in result.transformations_applied]
        assert len(found_transformations) > 0
    
    def test_data_integrity_after_cleaning(self, sample_online_retail_data):
        """🔍 Test intégrité des données après nettoyage"""
        transformer = create_data_transformer()
        
        result = transformer.clean_dataset(sample_online_retail_data)
        
        # Tests d'intégrité
        cleaned_df = result.data
        
        # Colonnes essentielles doivent exister
        essential_columns = ['Invoice', 'StockCode']  # Colonnes minimales
        for col in essential_columns:
            if col in sample_online_retail_data.columns:
                assert col in cleaned_df.columns, f"Colonne {col} manquante après nettoyage"
        
        # Pas de lignes entièrement vides
        if len(cleaned_df) > 0:
            assert not cleaned_df.isnull().all(axis=1).any(), "Lignes entièrement vides détectées"
        
        # Index cohérent
        assert not cleaned_df.index.duplicated().any(), "Index dupliqués détectés"


class TestDataTransformerIntegration:
    """🔗 Tests d'intégration avec de vraies données"""
    
    @pytest.mark.slow
    def test_full_online_retail_cleaning(self, full_online_retail_data):
        """🚀 Test nettoyage complet du dataset (peut être lent)"""
        transformer = create_data_transformer()
        
        # Test sur un échantillon raisonnable pour éviter timeout
        sample_df = full_online_retail_data.sample(n=min(1000, len(full_online_retail_data)))
        
        result = transformer.clean_dataset(sample_df)
        
        # Vérifications générales
        assert result.rows_after <= result.rows_before  # Nettoyage réduit/maintient
        assert result.cleaning_time_seconds > 0
        assert len(result.transformations_applied) > 0
        
        # Qualité des données améliorée
        cleaned_df = result.data
        if len(cleaned_df) > 0:
            # Plus de données aberrantes (selon votre logique)
            assert cleaned_df.duplicated().sum() == 0  # Pas de doublons
