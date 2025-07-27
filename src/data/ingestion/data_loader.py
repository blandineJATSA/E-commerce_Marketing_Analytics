# src/data/data_loader.py
"""
📁 Data Loader - UNIQUEMENT chargement fichiers
🎯 Single Responsibility: Load raw data from CSV
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from config.settings import settings, get_logger

logger = get_logger(__name__)

@dataclass
class RawDataResult:
    """📊 Résultat de chargement RAW (sans nettoyage)"""
    data: pd.DataFrame
    filename: str
    rows_count: int
    columns_count: int
    file_size_mb: float
    loading_time_seconds: float
    encoding_used: str

class EcommerceDataLoader:
    """
    📁 Chargeur de données CSV - RESPONSABILITÉ UNIQUE
    
    ✅ Ce qu'il fait:
    - Charge fichiers CSV 
    - Gère encodage automatique
    - Détecte délimiteurs
    - Cache en mémoire
    - Log métriques de chargement
    
    ❌ Ce qu'il ne fait PAS:
    - Validation qualité → data_validator.py 
    - Nettoyage → data_transformer.py
    - Business logic → features/
    """
    
    def __init__(self, enable_cache: bool = True):
        self.raw_data_dir = settings.RAW_DATA_DIR
        self.enable_cache = enable_cache
        self._cache = {}
        
        logger.info(f"📁 DataLoader initialisé: {self.raw_data_dir}")
    
    def load_csv(
        self,
        filename: str,
        encoding: Optional[str] = None,
        separator: Optional[str] = None,
        force_reload: bool = False
    ) -> RawDataResult:
        """
        📥 Charge un fichier CSV brut (SANS nettoyage)
        
        Returns:
            RawDataResult avec données brutes non modifiées
        """
        start_time = datetime.now()
        
        # Cache check
        cache_key = f"{filename}_{encoding}_{separator}"
        if not force_reload and cache_key in self._cache:
            logger.info("⚡ Données en cache")
            return self._cache[cache_key]
        
        file_path = self.raw_data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"❌ Fichier non trouvé: {file_path}")
        
        # Auto-détection encoding si nécessaire
        encoding_used = encoding or self._detect_encoding(file_path)
        
        # Auto-détection separator si nécessaire  
        separator_used = separator or self._detect_separator(file_path, encoding_used)
        
        # Chargement CSV brut
        logger.info(f"📖 Chargement: {filename} (encoding={encoding_used})")
        df = pd.read_csv(
            file_path,
            encoding=encoding_used,
            sep=separator_used,
            low_memory=False
        )
        
        # Métriques
        loading_time = (datetime.now() - start_time).total_seconds()
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        
        result = RawDataResult(
            data=df,
            filename=filename,
            rows_count=len(df),
            columns_count=len(df.columns),
            file_size_mb=file_size_mb,
            loading_time_seconds=loading_time,
            encoding_used=encoding_used
        )
        
        # Cache
        if self.enable_cache:
            self._cache[cache_key] = result
        
        logger.info(f"✅ Chargé: {len(df):,} lignes × {len(df.columns)} cols en {loading_time:.2f}s")
        
        return result
    
    def _detect_encoding(self, file_path: Path) -> str:
        """🔍 Détection automatique encodage"""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                sample = f.read(10000)
                detection = chardet.detect(sample)
                return detection['encoding'] or 'utf-8'
        except ImportError:
            # Fallback sans chardet
            encodings_to_try = ['utf-8', 'iso-8859-1', 'cp1252']
            for enc in encodings_to_try:
                try:
                    pd.read_csv(file_path, encoding=enc, nrows=5)
                    return enc
                except UnicodeDecodeError:
                    continue
            return 'utf-8'
    
    def _detect_separator(self, file_path: Path, encoding: str) -> str:
        """🔍 Détection automatique séparateur"""
        with open(file_path, 'r', encoding=encoding) as f:
            first_line = f.readline()
            
        # Test séparateurs courants
        separators = [',', ';', '\t', '|']
        separator_counts = {sep: first_line.count(sep) for sep in separators}
        
        # Retourne celui le plus fréquent
        return max(separator_counts, key=separator_counts.get)
    
    def list_files(self) -> List[str]:
        """📋 Liste fichiers CSV disponibles"""
        return sorted([f.name for f in self.raw_data_dir.glob("*.csv")])
    
    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """📊 Info fichier (sans le charger entièrement)"""
        file_path = self.raw_data_dir / filename
        if not file_path.exists():
            return {"error": "Fichier non trouvé"}
        
        # Stats système
        stat = file_path.stat()
        
        # Preview (5 premières lignes)
        try:
            preview_df = pd.read_csv(file_path, nrows=5, encoding='utf-8')
            preview = {
                "columns": list(preview_df.columns),
                "sample_data": preview_df.to_dict('records')
            }
        except Exception as e:
            preview = {"error": str(e)}
        
        return {
            "filename": filename,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "preview": preview
        }

def create_data_loader() -> EcommerceDataLoader:
    """🏭 Factory function"""
    return EcommerceDataLoader()
