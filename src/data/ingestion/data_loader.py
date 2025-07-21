# src/data/ingestion/data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from typing import Optional, Dict, Any
from datetime import datetime

class DataLoader:
    """Classe pour charger et ingérer les données e-commerce"""
    
    def __init__(self, config_path: str = "config/pipeline_settings.yaml"):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.data_config = self.config['data']
        
    def _load_config(self, config_path: str) -> Dict[Any, Any]:
        """Charge la configuration du pipeline"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Charge les données brutes depuis Excel
        
        Args:
            file_path: Chemin vers le fichier (optionnel)
            
        Returns:
            DataFrame: Données chargées
        """
        if file_path is None:
            file_path = Path(self.data_config['raw_path']) / self.data_config['file_name']
        
        try:
            self.logger.info(f"🔄 Chargement des données depuis : {file_path}")
            
            # Chargement avec gestion des erreurs
            df = pd.read_excel(file_path, engine='openpyxl')
            
            self.logger.info(f"✅ Données chargées avec succès")
            self.logger.info(f"📊 Shape: {df.shape}")
            self.logger.info(f"📋 Colonnes: {list(df.columns)}")
            
            return df
            
        except FileNotFoundError:
            self.logger.error(f"❌ Fichier non trouvé : {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du chargement : {str(e)}")
            raise
    
    def load_multiple_files(self, file_patterns: list) -> pd.DataFrame:
        """
        Charge et combine plusieurs fichiers
        
        Args:
            file_patterns: Liste des patterns de fichiers
            
        Returns:
            DataFrame: Données combinées
        """
        dfs = []
        raw_path = Path(self.data_config['raw_path'])
        
        for pattern in file_patterns:
            files = list(raw_path.glob(pattern))
            self.logger.info(f"🔍 Trouvé {len(files)} fichiers pour pattern: {pattern}")
            
            for file in files:
                try:
                    df = pd.read_excel(file, engine='openpyxl')
                    df['source_file'] = file.name
                    dfs.append(df)
                    self.logger.info(f"✅ Chargé: {file.name} - Shape: {df.shape}")
                except Exception as e:
                    self.logger.error(f"❌ Erreur avec {file.name}: {str(e)}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            self.logger.info(f"🔗 Données combinées - Shape finale: {combined_df.shape}")
            return combined_df
        else:
            raise ValueError("Aucun fichier valide trouvé")
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, format: str = 'parquet'):
        """
        Sauvegarde les données traitées
        
        Args:
            df: DataFrame à sauvegarder
            filename: Nom du fichier
            format: Format de sauvegarde ('parquet', 'csv', 'feather')
        """
        processed_path = Path(self.data_config['processed_path'])
        processed_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_with_timestamp = f"{filename}_{timestamp}"
        
        if format == 'parquet':
            file_path = processed_path / f"{filename_with_timestamp}.parquet"
            df.to_parquet(file_path, index=False)
        elif format == 'csv':
            file_path = processed_path / f"{filename_with_timestamp}.csv"
            df.to_csv(file_path, index=False)
        elif format == 'feather':
            file_path = processed_path / f"{filename_with_timestamp}.feather"
            df.to_feather(file_path)
        else:
            raise ValueError(f"Format non supporté: {format}")
        
        self.logger.info(f"💾 Données sauvegardées: {file_path}")
        return file_path
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Retourne des informations sur le dataset
        
        Args:
            df: DataFrame à analyser
            
        Returns:
            Dict: Informations du dataset
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'null_counts': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'sample_data': df.head(3).to_dict()
        }
        
        return info
