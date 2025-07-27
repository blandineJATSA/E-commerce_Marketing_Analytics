# tests/conftest.py
"""Fixtures utilisant les VRAIES données dans data/raw/"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os

# 🎯 CHEMIN VERS VOS VRAIES DONNÉES
@pytest.fixture(scope="session")
def raw_data_dir():
    """📁 Chemin vers data/raw/ réel"""
    project_root = Path(__file__).parent.parent  # Remonte de tests/ vers racine
    return project_root / "data" / "raw"

@pytest.fixture
def online_retail_file(raw_data_dir):
    """📁 Fichier online_retail_II.csv réel"""
    file_path = raw_data_dir / "online_retail_II.csv"
    if not file_path.exists():
        pytest.skip(f"Fichier {file_path} non trouvé - tests skippés")
    return file_path

@pytest.fixture
def sample_online_retail_data(online_retail_file):
    """🧪 Échantillon de vos vraies données (100 premières lignes)"""
    return pd.read_csv(online_retail_file, nrows=100, encoding='utf-8')

@pytest.fixture
def full_online_retail_data(online_retail_file):
    """📊 Données complètes (utilisé avec parcimonie)"""
    return pd.read_csv(online_retail_file, encoding='utf-8')

# 🧪 FIXTURES POUR CAS SPÉCIAUX (créées temporairement)
@pytest.fixture
def corrupted_csv_file():
    """📁 Fichier CSV corrompu pour tests d'erreur"""
    content = """Invoice,StockCode,Price
536365,85123A,invalid_price
536366,malformed_line"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield Path(temp_path)
    os.unlink(temp_path)

@pytest.fixture
def empty_csv_file():
    """📁 Fichier CSV vide pour tests edge case"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("")  # Fichier vide
        temp_path = f.name
    
    yield Path(temp_path)
    os.unlink(temp_path)

# 🛠️ FIXTURES POUR TRANSFORMER (basées sur vos vraies colonnes)
@pytest.fixture
def sample_ecommerce_raw_data(sample_online_retail_data):
    """🧪 Données brutes pour transformer (sous-ensemble des vraies données)"""
    # Prendre un échantillon et ajouter quelques cas de test
    df = sample_online_retail_data.head(20).copy()
    
    # Ajouter quelques cas problématiques pour tester
    if len(df) > 5:
        df.loc[df.index[0], 'Customer ID'] = None  # Valeur manquante
        df.loc[df.index[1], 'Description'] = None  # Description manquante
    
    return df

@pytest.fixture
def data_with_duplicates(sample_online_retail_data):
    """🧪 Données avec doublons ajoutés"""
    df = sample_online_retail_data.head(10).copy()
    
    # Dupliquer les 3 premières lignes
    duplicates = df.head(3).copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df


# tests/conftest.py - AJOUTS pour data_transformer

@pytest.fixture
def data_with_missing_values():
    """🧪 Données avec valeurs manquantes typiques e-commerce"""
    return pd.DataFrame({
        'Invoice': ['536365', '536366', None, '536368'],
        'StockCode': ['85123A', None, '71053', '84029G'],
        'Description': ['WHITE HANGING HEART', 'CREAM CUPID', None, 'KNITTED UNION'],
        'Quantity': [6, 1, None, 6],
        'InvoiceDate': ['12/1/2010 8:26', None, '12/1/2010 8:28', '12/1/2010 8:34'],
        'Price': [2.55, None, 3.39, 3.75],
        'Customer ID': [17850.0, None, 17850.0, 13047.0],
        'Country': ['United Kingdom', 'United Kingdom', None, 'United Kingdom']
    })

@pytest.fixture
def data_with_duplicates():
    """🧪 Données avec doublons complets"""
    base_data = pd.DataFrame({
        'Invoice': ['536365', '536366'],
        'StockCode': ['85123A', '71053'],
        'Description': ['WHITE HANGING HEART', 'WHITE METAL LANTERN'],
        'Quantity': [6, 6],
        'InvoiceDate': ['12/1/2010 8:26', '12/1/2010 8:26'],
        'Price': [2.55, 3.39],
        'Customer ID': [17850.0, 17850.0],
        'Country': ['United Kingdom', 'United Kingdom']
    })
    
    # Dupliquer les lignes
    duplicated_data = pd.concat([base_data, base_data, base_data], ignore_index=True)
    return duplicated_data

@pytest.fixture
def data_with_returns():
    """🧪 Données avec retours (Invoice C)"""
    return pd.DataFrame({
        'Invoice': ['536365', 'C536366', '536367', 'C536368'],  # 2 retours
        'StockCode': ['85123A', '85123A', '71053', '71053'],
        'Description': ['WHITE HANGING HEART', 'WHITE HANGING HEART', 'LANTERN', 'LANTERN'],
        'Quantity': [6, -6, 1, -1],  # Quantités négatives pour retours
        'InvoiceDate': ['12/1/2010 8:26', '12/1/2010 9:26', '12/1/2010 8:30', '12/1/2010 9:30'],
        'Price': [2.55, 2.55, 3.39, 3.39],
        'Customer ID': [17850.0, 17850.0, 17851.0, 17851.0],
        'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom']
    })
