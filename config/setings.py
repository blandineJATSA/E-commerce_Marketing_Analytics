# config/settings.py
"""
⚙️ Configuration centralisée du projet E-commerce Marketing Analytics
🎯 Optimisé pour fichiers CSV + MLOps stack
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, validator
import logging

class Settings(BaseSettings):
    """
    🔧 Configuration principale du projet
    Variables d'environnement + valeurs par défaut
    """
    
    # ==========================================
    # 📁 CHEMINS ET STRUCTURE PROJET
    # ==========================================
    
    # Racine du projet
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    
    # Dossiers principaux
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
    
    # Dossiers code
    SRC_DIR: Path = PROJECT_ROOT / "src"
    NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
    
    # Outputs
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # ==========================================
    # 📊 CONFIGURATION DONNÉES CSV
    # ==========================================
    
    # Fichiers CSV principaux (personnalisez selon vos fichiers)
    MAIN_ECOMMERCE_FILE: str = "ecommerce_data.csv"
    CUSTOMERS_FILE: str = "customers.csv" 
    PRODUCTS_FILE: str = "products.csv"
    ORDERS_FILE: str = "orders.csv"
    
    # Colonnes obligatoires attendues (validation)
    REQUIRED_COLUMNS: List[str] = [
        "customer_id", 
        "order_date", 
        "total_amount"
    ]
    
    # Formats de dates acceptés
    DATE_FORMATS: List[str] = [
        "%Y-%m-%d",
        "%d/%m/%Y", 
        "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S"
    ]
    
    # ==========================================
    # 🗃️ BASE DE DONNÉES (Optionnel - si upgrade)
    # ==========================================
    
    # PostgreSQL (pour plus tard si migration CSV → DB)
    DATABASE_URL: Optional[str] = None
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "ecommerce_analytics"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "password"
    
    # ==========================================
    # 🧠 MLFLOW CONFIGURATION
    # ==========================================
    
    MLFLOW_TRACKING_URI: str = "file://./mlflow/mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "ecommerce-marketing-analytics"
    MLFLOW_ARTIFACT_ROOT: str = "./mlflow/artifacts"
    MLFLOW_REGISTRY_URI: Optional[str] = None
    
    # ==========================================
    # 🔴 REDIS (Cache & Features)
    # ==========================================
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_URL: Optional[str] = None
    
    # Cache TTL (Time To Live)
    CACHE_TTL_HOURS: int = 24
    FEATURES_CACHE_TTL: int = 3600  # 1 heure pour features
    
    # ==========================================
    # 🚀 API FASTAPI
    # ==========================================
    
    API_TITLE: str = "E-commerce Marketing Analytics API"
    API_DESCRIPTION: str = "🎯 API MLOps pour segmentation clients et optimisation campagnes"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = True
    
    # Sécurité API
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # ==========================================
    # 🌬️ AIRFLOW CONFIGURATION  
    # ==========================================
    
    AIRFLOW_HOME: str = str(PROJECT_ROOT / "airflow")
    AIRFLOW_DAGS_DIR: str = str(PROJECT_ROOT / "pipelines" / "airflow_dags")
    
    # ==========================================
    # 📊 BUSINESS LOGIC PARAMETERS
    # ==========================================
    
    # RFM Segmentation
    RFM_QUANTILES: int = 5  # Quintiles pour RFM
    RFM_REFERENCE_DATE: Optional[str] = None  # Auto = max date
    
    # Clustering
    DEFAULT_N_CLUSTERS: int = 5
    MIN_CLUSTER_SIZE: int = 100
    
    # Campaign ROI
    MIN_CAMPAIGN_BUDGET: float = 1000.0
    MAX_CAMPAIGN_BUDGET: float = 100000.0
    DEFAULT_CONVERSION_RATE: float = 0.02  # 2%
    
    # ==========================================
    # 🔧 LOGGING CONFIGURATION
    # ==========================================
    
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "ecommerce_analytics.log"
    
    # ==========================================
    # 🐳 ENVIRONMENT SETTINGS
    # ==========================================
    
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = True
    TESTING: bool = False
    
    # ==========================================
    # 📈 MONITORING & METRICS
    # ==========================================
    
    # Prometheus metrics
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Alerting
    ENABLE_ALERTS: bool = False
    SLACK_WEBHOOK_URL: Optional[str] = None
    EMAIL_ALERTS: List[str] = []
    
    # ==========================================
    # 🔧 VALIDATORS & POST-INIT
    # ==========================================
    
    @validator('DATABASE_URL', pre=True, always=True)
    def build_database_url(cls, v, values):
        """🔗 Construction URL database"""
        if v:
            return v
        return (
            f"postgresql://{values.get('DB_USER')}:{values.get('DB_PASSWORD')}"
            f"@{values.get('DB_HOST')}:{values.get('DB_PORT')}/{values.get('DB_NAME')}"
        )
    
    @validator('REDIS_URL', pre=True, always=True) 
    def build_redis_url(cls, v, values):
        """🔗 Construction URL Redis"""
        if v:
            return v
        password_part = f":{values.get('REDIS_PASSWORD')}@" if values.get('REDIS_PASSWORD') else ""
        return f"redis://{password_part}{values.get('REDIS_HOST')}:{values.get('REDIS_PORT')}/{values.get('REDIS_DB')}"
    
    def create_directories(self):
        """📁 Création des dossiers nécessaires"""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR, 
            self.PROCESSED_DATA_DIR,
            self.EXTERNAL_DATA_DIR,
            self.MODELS_DIR,
            self.OUTPUTS_DIR,
            self.LOGS_DIR,
            Path(self.MLFLOW_TRACKING_URI.replace('file://', '')).parent,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logging.info(f"📁 Dossiers créés: {len(directories)} dossiers")
    
    def get_data_file_path(self, filename: str, data_type: str = "raw") -> Path:
        """🎯 Helper pour chemins fichiers données"""
        data_dirs = {
            "raw": self.RAW_DATA_DIR,
            "processed": self.PROCESSED_DATA_DIR, 
            "external": self.EXTERNAL_DATA_DIR
        }
        return data_dirs[data_type] / filename
    
    def is_production(self) -> bool:
        """🚀 Check si environnement production"""
        return self.ENVIRONMENT.lower() == "production"
    
    def is_development(self) -> bool:
        """🛠️ Check si environnement development"""  
        return self.ENVIRONMENT.lower() == "development"
    
    class Config:
        """⚙️ Configuration Pydantic"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# ==========================================
# 🏭 INSTANCE GLOBALE
# ==========================================

# Instance unique de configuration
settings = Settings()

# Création automatique des dossiers au import
settings.create_directories()

# ==========================================
# 🎯 HELPERS & UTILITIES
# ==========================================

def get_logger(name: str) -> logging.Logger:
    """📝 Helper pour créer logger configuré"""
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Handler console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
        
        # Handler fichier
        file_handler = logging.FileHandler(
            settings.LOGS_DIR / settings.LOG_FILE
        )
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(settings.LOG_FORMAT)
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Ajout handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    return logger

def load_config_from_env() -> Dict[str, Any]:
    """🔧 Charge config depuis variables d'environnement"""
    return settings.dict()

# ==========================================
# 📋 EXPORT CONFIGURATION
# ==========================================

__all__ = [
    "settings",
    "Settings", 
    "get_logger",
    "load_config_from_env"
]
