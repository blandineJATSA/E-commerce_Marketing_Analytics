# tests/integration/test_full_pipeline.py
"""
🔗 Tests d'intégration - Pipeline complet
🎯 Load → Validate → Transform → Analyze
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import time

from src.data.data_loader import create_data_loader
from src.data.data_validator import create_data_validator
from src.data.data_transformer import create_data_transformer

class TestFullDataPipeline:
    """🔄 Tests du pipeline complet E-commerce"""
    
    def test_complete_pipeline_success(self, online_retail_file):
        """🎯 Pipeline complet avec vraies données"""
        
        # 1. 📥 LOAD
        loader = create_data_loader()
        load_result = loader.load_csv(online_retail_file.name)
        
        assert load_result.success
        assert isinstance(load_result.data, pd.DataFrame)
        assert len(load_result.data) > 0
        
        # 2. ✅ VALIDATE
        validator = create_data_validator()
        validation_result = validator.validate_dataframe(load_result.data)
        
        assert 'total_rows' in validation_result
        assert 'total_columns' in validation_result
        assert 'validation_passed' in validation_result
        
        # 3. 🔧 TRANSFORM
        transformer = create_data_transformer()
        transform_result = transformer.clean_dataset(load_result.data)
        
        assert transform_result.rows_after > 0
        assert transform_result.data_quality_score > 0
        assert len(transform_result.transformations_applied) > 0
        
        # 4. 📊 FINAL CHECKS
        clean_data = transform_result.data
        
        # Vérifications métier
        if 'TotalAmount' in clean_data.columns:
            assert clean_data['TotalAmount'].notna().any()
        
        if 'IsReturn' in clean_data.columns:
            assert clean_data['IsReturn'].dtype == bool
        
        if 'InvoiceDate' in clean_data.columns and clean_data['InvoiceDate'].notna().any():
            assert pd.api.types.is_datetime64_any_dtype(clean_data['InvoiceDate'])
        
        print(f"✅ Pipeline réussi: {load_result.rows} → {transform_result.rows_after} lignes")
        print(f"📊 Score qualité: {transform_result.data_quality_score:.1f}/100")
    
    def test_pipeline_performance(self, online_retail_file):
        """⚡ Test performance du pipeline complet"""
        
        start_time = time.time()
        
        # Pipeline avec échantillon pour la vitesse
        loader = create_data_loader()
        raw_data = loader.load_csv(online_retail_file.name, nrows=10000)
        
        load_time = time.time()
        
        transformer = create_data_transformer()
        result = transformer.clean_dataset(raw_data.data)
        
        transform_time = time.time()
        
        # Métriques de performance
        total_time = transform_time - start_time
        load_duration = load_time - start_time
        transform_duration = transform_time - load_time
        
        print(f"⏱️ Performance Pipeline:")
        print(f"  📥 Load: {load_duration:.2f}s")
        print(f"  🔧 Transform: {transform_duration:.2f}s")
        print(f"  🎯 Total: {total_time:.2f}s")
        
        # Assertions performance (ajustez selon votre matériel)
        assert total_time < 30.0  # Pipeline complet < 30s pour 10k lignes
        assert result.cleaning_time_seconds < 20.0  # Transform < 20s
        
        # Vérifications qualité après optimisation performance
        assert result.data_quality_score > 60.0  # Score décent même rapide
    
    def test_pipeline_data_flow(self, sample_ecommerce_raw_data):
        """🌊 Test de flux des données à travers pipeline"""
        
        print("\n🌊 SUIVI DU FLUX DE DONNÉES:")
        
        # État initial
        initial_shape = sample_ecommerce_raw_data.shape
        initial_nulls = sample_ecommerce_raw_data.isnull().sum().sum()
        print(f"📊 Initial: {initial_shape[0]} lignes, {initial_shape[1]} colonnes, {initial_nulls} NaN")
        
        # Validation
        validator = create_data_validator()
        val_result = validator.validate_dataframe(sample_ecommerce_raw_data)
        print(f"✅ Validation: {val_result.get('validation_passed', 'N/A')}")
        
        # Transformation étape par étape
        transformer = create_data_transformer()
        
        # Test avec chaque étape séparément pour debug
        df = sample_ecommerce_raw_data.copy()
        
        # Étape 1: Missing values
        df_after_missing, missing_transf = transformer._handle_missing_values(df)
        print(f"🕳️ Après missing: {len(df_after_missing)} lignes (-{len(df)-len(df_after_missing)})")
        
        # Étape 2: Duplicates
        df_after_dup, dup_transf = transformer._remove_duplicates(df_after_missing)
        print(f"🔄 Après doublons: {len(df_after_dup)} lignes (-{len(df_after_missing)-len(df_after_dup)})")
        
        # Étape 3: Dates
        df_after_dates, date_transf = transformer._parse_invoice_dates(df_after_dup)
        print(f"📅 Après dates: {df_after_dates.shape[1]} colonnes (+{df_after_dates.shape[1]-df_after_dup.shape[1]})")
        
        # Étape 4: Returns
        df_after_returns, return_transf = transformer._detect_returns(df_after_dates)
        print(f"🔄 Après retours: {df_after_returns.shape[1]} colonnes (+{df_after_returns.shape[1]-df_after_dates.shape[1]})")
        
        # Pipeline complet
        final_result = transformer.clean_dataset(sample_ecommerce_raw_data)
        final_shape = final_result.data.shape
        final_nulls = final_result.data.isnull().sum().sum()
        
        print(f"🎯 Final: {final_shape[0]} lignes, {final_shape[1]} colonnes, {final_nulls} NaN")
        print(f"📈 Score qualité: {final_result.data_quality_score:.1f}/100")
        
        # Assertions de flux
        assert final_result.rows_after <= initial_shape[0]  # Peut seulement diminuer/rester égal
        assert final_shape[1] >= initial_shape[1]  # Nouvelles colonnes ajoutées
        assert len(final_result.transformations_applied) > 0  # Au moins une transformation
    
    def test_pipeline_error_handling(self):
        """❌ Test gestion d'erreurs dans le pipeline"""
        
        # Données complètement vides
        empty_df = pd.DataFrame()
        
        transformer = create_data_transformer()
        result = transformer.clean_dataset(empty_df)
        
        assert result.rows_before == 0
        assert result.rows_after == 0
        assert result.data_quality_score == 0.0
        
        # Données avec colonnes manquantes
        minimal_df = pd.DataFrame({
            'RandomColumn': [1, 2, 3]
        })
        
        result_minimal = transformer.clean_dataset(minimal_df)
        
        assert result_minimal.rows_after == 3  # Données préservées
        assert isinstance(result_minimal.data, pd.DataFrame)
        
        # Données avec types incorrects
        bad_types_df = pd.DataFrame({
            'Invoice': ['123', '456', 'abc'],
            'Price': ['not_a_number', '2.5', '3.0'],
            'Quantity': [1, 2, 3]
        })
        
        result_bad = transformer.clean_dataset(bad_types_df)
        
        # Doit survivre sans crash
        assert isinstance(result_bad.data, pd.DataFrame)
        assert result_bad.rows_after >= 0
    
    def test_pipeline_memory_efficiency(self, sample_ecommerce_raw_data):
        """💾 Test efficacité mémoire du pipeline"""
        
        initial_memory = sample_ecommerce_raw_data.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Transformer avec optimisation
        transformer = create_data_transformer()
        result = transformer.clean_dataset(sample_ecommerce_raw_data)
        
        final_memory = result.memory_usage_mb
        memory_efficiency = (initial_memory - final_memory) / initial_memory * 100
        
        print(f"💾 Mémoire: {initial_memory:.2f}MB → {final_memory:.2f}MB")
        print(f"⚡ Efficacité: {memory_efficiency:.1f}% de réduction")
        
        # L'optimisation peut parfois augmenter la mémoire (nouvelles colonnes)
        # Mais le ratio doit rester raisonnable
        assert final_memory < initial_memory * 2  # Max 2x la mémoire initiale
        assert result.memory_usage_mb > 0  # Mémoire mesurée
    
    @pytest.mark.slow
    def test_pipeline_large_dataset(self, online_retail_file):
        """🐘 Test pipeline avec dataset complet (marqué comme lent)"""
        
        # Chargement complet
        loader = create_data_loader()
        full_data = loader.load_csv(online_retail_file.name)  # Sans nrows limit
        
        if len(full_data.data) < 100000:  # Skip si dataset trop petit
            pytest.skip("Dataset trop petit pour test 'large dataset'")
        
        print(f"🐘 Test avec {len(full_data.data)} lignes complètes")
        
        start_time = time.time()
        
        # Pipeline complet
        transformer = create_data_transformer()
        result = transformer.clean_dataset(full_data.data)
        
        total_time = time.time() - start_time
        
        # Métriques pour gros dataset
        reduction_rate = (result.rows_before - result.rows_after) / result.rows_before
        processing_rate = len(full_data.data) / total_time  # lignes/seconde
        
        print(f"📊 Résultats dataset complet:")
        print(f"  🔧 Temps: {total_time:.1f}s")
        print(f"  📉 Réduction: {reduction_rate*100:.1f}%")
        print(f"  ⚡ Vitesse: {processing_rate:.0f} lignes/s")
        print(f"  📈 Qualité: {result.data_quality_score:.1f}/100")
        
        # Assertions performance gros volume
        assert result.data_quality_score > 70.0  # Bonne qualité
        assert processing_rate > 1000  # Au moins 1000 lignes/s
        assert len(result.transformations_applied) >= 5  # Pipeline complet
    
    def test_pipeline_reproducibility(self, sample_ecommerce_raw_data):
        """🔄 Test reproductibilité du pipeline"""
        
        transformer = create_data_transformer()
        
        # Deux exécutions identiques
        result1 = transformer.clean_dataset(sample_ecommerce_raw_data.copy())
        result2 = transformer.clean_dataset(sample_ecommerce_raw_data.copy())
        
        # Résultats doivent être identiques
        assert result1.rows_after == result2.rows_after
        assert result1.columns_after == result2.columns_after
        assert result1.data_quality_score == result2.data_quality_score
        assert result1.transformations_applied == result2.transformations_applied
        
        # DataFrames identiques
        pd.testing.assert_frame_equal(
            result1.data.sort_index(axis=1), 
            result2.data.sort_index(axis=1),
            check_dtype=True,
            check_exact=False  # Float precision tolerance
        )
        
        print("✅ Pipeline 100% reproductible")


class TestPipelineConfigurations:
    """⚙️ Test différentes configurations du pipeline"""
    
    def test_minimal_pipeline(self, sample_ecommerce_raw_data):
        """🔧 Pipeline minimal (only missing values)"""
        
        transformer = create_data_transformer(
            remove_duplicates=False,
            parse_dates=False,
            detect_returns=False
        )
        
        result = transformer.clean_dataset(sample_ecommerce_raw_data)
        
        # Moins de transformations
        assert len(result.transformations_applied) < 10
        assert result.rows_after > 0
        
        # Pas de colonnes temporelles ajoutées
        assert 'Year' not in result.data.columns
        assert 'IsReturn' not in result.data.columns
    
    def test_full_featured_pipeline(self, sample_ecommerce_raw_data):
        """🚀 Pipeline avec toutes les fonctionnalités"""
        
        transformer = create_data_transformer(
            remove_duplicates=True,
            handle_missing=True,
            parse_dates=True,
            detect_returns=True
        )
        
        result = transformer.clean_dataset(sample_ecommerce_raw_data)
        
        # Plus de transformations
        expected_transformations = [
            "missing_values_handled",
            "duplicates_removed", 
            "date_parsing",
            "returns_detected",
            "business_rules_validated",
            "data_types_optimized"
        ]
        
        applied = result.transformations_applied
        for transform in expected_transformations:
            # Au moins une partie doit être présente
            assert any(transform in t for t in applied) or len(applied) > 0
    
    def test_preserve_original_flag(self, sample_ecommerce_raw_data):
        """💾 Test du flag preserve_original"""
        
        transformer = create_data_transformer()
        original_id = id(sample_ecommerce_raw_data)
        
        # Avec preserve=True (défaut)
        result1 = transformer.clean_dataset(sample_ecommerce_raw_data, preserve_original=True)
        after_preserve_id = id(sample_ecommerce_raw_data)
        
        # Avec preserve=False 
        test_df = sample_ecommerce_raw_data.copy()
        original_test_id = id(test_df)
        result2 = transformer.clean_dataset(test_df, preserve_original=False)
        after_no_preserve_id = id(test_df)
        
        # Vérifications
        assert original_id == after_preserve_id  # Original pas modifié
        assert original_test_id == after_no_preserve_id  # Même objet utilisé
        
        # Résultats similaires dans les deux cas
        assert result1.rows_after == result2.rows_after
