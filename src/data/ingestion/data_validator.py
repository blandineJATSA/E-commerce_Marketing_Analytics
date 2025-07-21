# src/data/ingestion/data_validator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import yaml

class DataValidator:
    """Classe pour valider la qualité et l'intégrité des données"""
    
    def __init__(self, config_path: str = "config/pipeline_settings.yaml"):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        
    def _load_config(self, config_path: str) -> Dict[Any, Any]:
        """Charge la configuration"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict[str, str]) -> bool:
        """
        Valide que les colonnes correspondent au schéma attendu
        
        Args:
            df: DataFrame à valider
            expected_schema: Schéma attendu {colonne: type}
            
        Returns:
            bool: True si valide
        """
        validation_passed = True
        issues = []
        
        # Vérifier les colonnes manquantes
        missing_columns = set(expected_schema.keys()) - set(df.columns)
        if missing_columns:
            issues.append(f"Colonnes manquantes: {missing_columns}")
            validation_passed = False
        
        # Vérifier les colonnes supplémentaires
        extra_columns = set(df.columns) - set(expected_schema.keys())
        if extra_columns:
            issues.append(f"Colonnes supplémentaires: {extra_columns}")
            self.logger.warning(f"⚠️ Colonnes non attendues: {extra_columns}")
        
        # Vérifier les types de données
        for col, expected_type in expected_schema.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not self._is_compatible_type(actual_type, expected_type):
                    issues.append(f"Type incorrect pour {col}: attendu {expected_type}, reçu {actual_type}")
                    validation_passed = False
        
        self.validation_results['schema_validation'] = {
            'passed': validation_passed,
            'issues': issues,
            'timestamp': datetime.now().isoformat()
        }
        
        if validation_passed:
            self.logger.info("✅ Validation du schéma réussie")
        else:
            self.logger.error(f"❌ Validation du schéma échouée: {issues}")
        
        return validation_passed
    
    def _is_compatible_type(self, actual_type: str, expected_type: str) -> bool:
        """Vérifie la compatibilité des types"""
        type_mapping = {
            'int64': ['int', 'integer', 'int64'],
            'float64': ['float', 'numeric', 'float64'],
            'object': ['string', 'object', 'text'],
            'datetime64[ns]': ['datetime', 'timestamp', 'datetime64[ns]'],
            'bool': ['boolean', 'bool']
        }
        
        if actual_type in type_mapping:
            return expected_type.lower() in type_mapping[actual_type]
        return False
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Effectue une validation complète de la qualité des données
        
        Args:
            df: DataFrame à valider
            
        Returns:
            Dict: Résultats de validation
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'null_analysis': self._analyze_null_values(df),
            'duplicate_analysis': self._analyze_duplicates(df),
            'data_types_analysis': self._analyze_data_types(df),
            'outlier_analysis': self._analyze_outliers(df),
            'consistency_checks': self._check_consistency(df)
        }
        
        # Calculer le score de qualité global
        quality_score = self._calculate_quality_score(quality_report)
        quality_report['quality_score'] = quality_score
        
        self.validation_results['data_quality'] = quality_report
        
        self.logger.info(f"📊 Score de qualité des données: {quality_score:.2f}/100")
        
        return quality_report
    
    def _analyze_null_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse les valeurs manquantes"""
        null_counts = df.isnull().sum()
        null_percentages = (null_counts / len(df)) * 100
        
        problematic_columns = null_percentages[null_percentages > 50].to_dict()
        
        return {
            'null_counts': null_counts.to_dict(),
            'null_percentages': null_percentages.to_dict(),
            'problematic_columns': problematic_columns,
            'total_null_values': null_counts.sum()
        }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse les doublons"""
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(df)) * 100
        
        return {
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': duplicate_percentage,
            'unique_rows': len(df) - duplicate_rows
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse les types de données"""
        dtype_counts = df.dtypes.value_counts().to_dict()
        
        return {
            'data_types': df.dtypes.to_dict(),
            'type_distribution': {str(k): v for k, v in dtype_counts.items()}
        }
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse les valeurs aberrantes pour les colonnes numériques"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_analysis = {}
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            outlier_analysis[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_value': df[col].min(),
                'max_value': df[col].max()
            }
        
        return outlier_analysis
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Vérifie la cohérence des données (spécifique e-commerce)"""
        consistency_issues = []
        
        # Vérifications spécifiques pour données e-commerce
        if 'Quantity' in df.columns:
            negative_quantities = (df['Quantity'] < 0).sum()
            if negative_quantities > 0:
                consistency_issues.append(f"Quantités négatives: {negative_quantities}")
        
        if 'UnitPrice' in df.columns:
            negative_prices = (df['UnitPrice'] < 0).sum()
            if negative_prices > 0:
                consistency_issues.append(f"Prix négatifs: {negative_prices}")
        
        if 'CustomerID' in df.columns:
            invalid_customer_ids = df['CustomerID'].isna().sum()
            if invalid_customer_ids > 0:
                consistency_issues.append(f"CustomerID manquants: {invalid_customer_ids}")
        
        return {
            'consistency_issues': consistency_issues,
            'issues_count': len(consistency_issues)
        }
    
    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calcule un score de qualité global"""
        score = 100.0
        
        # Pénalités pour valeurs manquantes
        avg_null_percentage = np.mean(list(quality_report['null_analysis']['null_percentages'].values()))
        score -= avg_null_percentage * 0.5
        
        # Pénalités pour doublons
        duplicate_penalty = quality_report['duplicate_analysis']['duplicate_percentage'] * 0.3
        score -= duplicate_penalty
        
        # Pénalités pour problèmes de cohérence
        consistency_penalty = quality_report['consistency_checks']['issues_count'] * 10
        score -= consistency_penalty
        
        return max(0, min(100, score))
    
    def generate_validation_report(self) -> str:
        """Génère un rapport de validation détaillé"""
        if not self.validation_results:
            return "Aucune validation effectuée"
        
        report = []
        report.append("=" * 50)
        report.append("RAPPORT DE VALIDATION DES DONNÉES")
        report.append("=" * 50)
        
        for validation_type, results in self.validation_results.items():
            report.append(f"\n{validation_type.upper()}:")
            report.append("-" * 30)
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict):
                        report.append(f"{key}:")
                        for sub_key, sub_value in value.items():
                            report.append(f"  {sub_key}: {sub_value}")
                    else:
                        report.append(f"{key}: {value}")
        
        return "\n".join(report)
