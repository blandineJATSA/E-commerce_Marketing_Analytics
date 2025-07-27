# tests/unit/test_data_loader.py - VERSION CORRIGÉE
"""🧪 Tests utilisant les VRAIES données dans data/raw/"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.data.data_loader import create_data_loader, RawDataResult

class TestEcommerceDataLoader:
    
    def test_load_real_online_retail_csv(self, raw_data_dir, online_retail_file):
        """✅ Test chargement du VRAI fichier online_retail_II.csv"""
        loader = create_data_loader()
        
        with patch.object(loader, 'raw_data_dir', raw_data_dir):
            result = loader.load_csv("online_retail_II.csv")
        
        # Vérifications avec VRAIES données
        assert isinstance(result, RawDataResult)
        assert result.rows_count > 1000  # On sait qu'il y a beaucoup de lignes
        assert result.columns_count == 8  # Colonnes attendues
        assert 'Invoice' in result.data.columns
        assert 'Customer ID' in result.data.columns
        assert result.filename == "online_retail_II.csv"
    
    def test_list_files_real_directory(self, raw_data_dir):
        """📋 Test listage fichiers dans VRAI dossier data/raw/"""
        loader = create_data_loader()
        
        with patch.object(loader, 'raw_data_dir', raw_data_dir):
            files = loader.list_files()
        
        assert isinstance(files, list)
        assert "online_retail_II.csv" in files
    
    def test_get_file_info_real_file(self, raw_data_dir):
        """📊 Test info sur VRAI fichier"""
        loader = create_data_loader()
        
        with patch.object(loader, 'raw_data_dir', raw_data_dir):
            info = loader.get_file_info("online_retail_II.csv")
        
        assert info['filename'] == "online_retail_II.csv"
        assert info['size_mb'] > 0  # Le fichier a une taille
        assert 'preview' in info
        
        # Vérifier colonnes attendues dans le preview
        if 'columns' in info['preview']:
            expected_columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 
                              'InvoiceDate', 'Price', 'Customer ID', 'Country']
            assert all(col in info['preview']['columns'] for col in expected_columns)
    
    # Tests edge cases avec fichiers temporaires (quand même nécessaires)
    def test_load_csv_file_not_found(self, raw_data_dir):
        """❌ Test fichier non trouvé"""
        loader = create_data_loader()
        
        with patch.object(loader, 'raw_data_dir', raw_data_dir):
            with pytest.raises(FileNotFoundError):
                loader.load_csv("inexistant.csv")
    
    def test_load_corrupted_csv(self, corrupted_csv_file):
        """❌ Test fichier CSV corrompu"""
        loader = create_data_loader()
        
        with patch.object(loader, 'raw_data_dir', corrupted_csv_file.parent):
            # Doit charger même si données corrompues (pandas est tolérant)
            result = loader.load_csv(corrupted_csv_file.name)
            assert isinstance(result.data, pd.DataFrame)
