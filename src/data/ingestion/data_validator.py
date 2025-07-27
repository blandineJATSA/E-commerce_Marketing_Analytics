# src/data/data_validator.py
"""
🔍 Data Validator - UNIQUEMENT validation qualité
🎯 Single Responsibility: Validate data quality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from config.settings import settings, get_logger

logger = get_logger(__name__)

@dataclass
class ValidationResult:
    """🔍 Résultat de validation"""
    is_valid: bool
    quality_score: float  # 0-100
    errors: List[str]
    warnings: List[str]
    missing_columns: List[str]
    data_quality_metrics: Dict[str, Any]
    recommendations: List[str]

class DataValidator:
    """
    🔍 Validateur de qualité des données
    
    ✅ Responsabilités:
    - Validation schéma (colonnes obligatoires)
    - Détection outliers
    - Score de qualité
    - Analyse valeurs manquantes
    - Recommandations amélioration
    """
    
    def __init__(self):
        self.required_columns = settings.REQUIRED_COLUMNS
        logger.info("🔍 DataValidator initialisé")
    
    def validate_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str = "unknown"
    ) -> ValidationResult:
        """
        🔍 Validation complète d'un dataset
        
        Args:
            df: DataFrame à valider (données brutes)
            dataset_name: Nom du dataset pour logs
            
        Returns:
            ValidationResult avec score et recommandations
        """
        logger.info(f"🔍 Validation dataset: {dataset_name}")
        
        errors = []
        warnings = []
        recommendations = []
        
        # 1️⃣ Validation schéma
        missing_cols = self._validate_schema(df, errors, warnings)
        
        # 2️⃣ Validation intégrité données
        self._validate_data_integrity(df, errors, warnings, recommendations)
        
        # 3️⃣ Calcul métriques qualité
        quality_metrics = self._calculate_quality_metrics(df)
        
        # 4️⃣ Score global
        quality_score = self._calculate_quality_score(df, quality_metrics)
        
        # 5️⃣ Recommandations
        recommendations.extend(self._generate_recommendations(df, quality_metrics))
        
        is_valid = len(errors) == 0 and quality_score >= 60
        
        result = ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            errors=errors,
            warnings=warnings,
            missing_columns=missing_cols,
            data_quality_metrics=quality_metrics,
            recommendations=recommendations
        )
        
        self._log_validation_result(result, dataset_name)
        
        return result
    
    def _validate_schema(self, df: pd.DataFrame, errors: List, warnings: List) -> List[str]:
        """🏗️ Validation schéma (colonnes obligatoires)"""
        
        missing_required = set(self.required_columns) - set(df.columns)
        
        if missing_required:
            for col in missing_required:
                errors.append(f"Colonne obligatoire manquante: {col}")
        
        # Colonnes vides
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            warnings.append(f"Colonnes entièrement vides: {empty_columns}")
        
        return list(missing_required)
    
    def _validate_data_integrity(
        self, 
        df: pd.DataFrame, 
        errors: List, 
        warnings: List,
        recommendations: List
    ):
        """🔍 Validation intégrité des données"""
        
        # Dataset vide
        if len(df) == 0:
            errors.append("Dataset vide (0 lignes)")
            return
        
        # Lignes entièrement vides
        empty_rows_pct = (df.isnull().all(axis=1).sum() / len(df)) * 100
        if empty_rows_pct > 10:
            warnings.append(f"{empty_rows_pct:.1f}% lignes entièrement vides")
        
        # Taux de valeurs manquantes critique
        missing_pct = (df.isnull().sum().sum() / df.size) * 100
        if missing_pct > 50:
            errors.append(f"Trop de valeurs manquantes: {missing_pct:.1f}%")
        elif missing_pct > 20:
            warnings.append(f"Beaucoup de valeurs manquantes: {missing_pct:.1f}%")
        
        # Doublons
        duplicates_pct = (df.duplicated().sum() / len(df)) * 100
        if duplicates_pct > 20:
            warnings.append(f"Beaucoup de doublons: {duplicates_pct:.1f}%")
        
        # Validation types de données
        self._validate_data_types(df, warnings, recommendations)
    
    def _validate_data_types(
        self, 
        df: pd.DataFrame, 
        warnings: List, 
        recommendations: List
    ):
        """📊 Validation types de données"""
        
        for col in df.columns:
            # Détection colonnes numériques stockées en texte
            if df[col].dtype == 'object':
                # Test si contient des nombres
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    numeric_count = sum(str(val).replace('.', '').replace(',', '').replace('-', '').isdigit() 
                                      for val in sample)
                    numeric_pct = numeric_count / len(sample)
                    
                    if numeric_pct > 0.8:
                        recommendations.append(f"Colonne '{col}' semble numérique mais stockée en texte")
            
            # Détection colonnes de dates
            if df[col].dtype == 'object' and any(keyword in col.lower() 
                                               for keyword in ['date', 'time', 'created', 'updated']):
                recommendations.append(f"Colonne '{col}' semble être une date à convertir")
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """📊 Calcul métriques de qualité détaillées"""
        
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values_count": df.isnull().sum().sum(),
            "missing_values_pct": (df.isnull().sum().sum() / df.size) * 100,
            "duplicate_rows": df.duplicated().sum(),
            "duplicate_rows_pct": (df.duplicated().sum() / len(df)) * 100,
            "empty_rows": df.isnull().all(axis=1).sum(),
            "empty_columns": df.isnull().all().sum(),
            "unique_rows_pct": (df.drop_duplicates().shape[0] / len(df)) * 100,
            "data_types": df.dtypes.value_counts().to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }
    
    def _calculate_quality_score(self, df: pd.DataFrame, metrics: Dict) -> float:
        """🏆 Calcul score qualité (0-100)"""
        
        score = 100.0
        
        # Pénalités
        score -= metrics["missing_values_pct"] * 1.5  # -1.5 par % manquant
        score -= metrics["duplicate_rows_pct"] * 1.0  # -1.0 par % doublons
        score -= (metrics["empty_columns"] / metrics["total_columns"]) * 20  # -20 par colonne vide
        
        # Bonus
        if metrics["total_rows"] >= 1000:
            score += 10
        if metrics["total_columns"] >= 5:
            score += 5
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, df: pd.DataFrame, metrics: Dict) -> List[str]:
        """💡 Génération recommandations"""
        
        recommendations = []
        
        if metrics["missing_values_pct"] > 10:
            recommendations.append("Considérer imputation des valeurs manquantes")
        
        if metrics["duplicate_rows_pct"] > 5:
            recommendations.append("Supprimer les doublons identifiés")
        
        if metrics["empty_columns"] > 0:
            recommendations.append("Supprimer les colonnes entièrement vides")
        
        if metrics["memory_usage_mb"] > 100:
            recommendations.append("Optimiser les types de données pour réduire la mémoire")
        
        return recommendations
    
    def _log_validation_result(self, result: ValidationResult, dataset_name: str):
        """📝 Log résultat validation"""
        
        status = "✅ VALIDE" if result.is_valid else "❌ INVALIDE"
        logger.info(f"{status} | {dataset_name} | Score: {result.quality_score:.1f}/100")
        
        if result.errors:
            for error in result.errors:
                logger.error(f"❌ {error}")
        
        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"⚠️ {warning}")
    
    def quick_check(self, df: pd.DataFrame) -> bool:
        """⚡ Validation rapide (True/False)"""
        result = self.validate_dataset(df)
        return result.is_valid

def create_data_validator() -> DataValidator:
    """🏭 Factory function"""
    return DataValidator()
