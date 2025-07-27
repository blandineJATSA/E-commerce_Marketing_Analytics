"""
Base Model - Classe abstraite pour tous les modèles ML
Interface commune + Validation + Métriques + Sauvegarde
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union
import pickle
import joblib
import json
import logging
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
warnings.filterwarnings('ignore')

class BaseModel(ABC):
    """
    🎯 Classe de base pour tous les modèles ML
    Interface commune + Validation + Métriques
    """
    
    def __init__(self, model_name: str, model_type: str = 'classification', config: Dict = None):
        """
        Initialise le modèle de base
        
        Args:
            model_name: Nom unique du modèle
            model_type: 'classification', 'regression', 'clustering'
            config: Configuration personnalisée
        """
        self.model_name = model_name
        self.model_type = model_type
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
        # État du modèle
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.target_name = None
        self.preprocessing_pipeline = None
        
        # Métriques et historique
        self.training_history = []
        self.validation_scores = {}
        self.feature_importance = {}
        
        # Configuration paths
        self.models_dir = Path(self.config.get('models_dir', 'models/saved'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def build_model(self, **kwargs) -> Any:
        """
        🏗️ Construit l'architecture du modèle
        À implémenter dans chaque sous-classe
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame, fit_transform: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        🔧 Préprocessing des données
        À implémenter selon le type de modèle
        """
        pass
    
    def validate_data(self, data: pd.DataFrame, required_columns: List[str] = None) -> bool:
        """
        ✅ Validation des données d'entrée
        """
        try:
            if data.empty:
                raise ValueError("Dataset vide")
            
            if required_columns:
                missing_cols = set(required_columns) - set(data.columns)
                if missing_cols:
                    raise ValueError(f"Colonnes manquantes: {missing_cols}")
            
            # Vérification valeurs manquantes critiques
            critical_na_pct = data.isnull().sum() / len(data)
            if (critical_na_pct > 0.8).any():
                self.logger.warning("Colonnes avec >80% de valeurs manquantes détectées")
            
            # Vérification doublons
            duplicates = data.duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"{duplicates} doublons détectés")
            
            self.logger.info(f"✅ Validation réussie: {len(data)} lignes, {len(data.columns)} colonnes")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Validation échouée: {e}")
            return False
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray] = None, 
              validation_split: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        🎓 Entraînement du modèle avec validation
        """
        try:
            self.logger.info(f"🎓 Début entraînement {self.model_name}...")
            
            # Conversion en DataFrame si nécessaire
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            
            # Split train/validation si demandé
            if validation_split > 0 and y is not None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split, random_state=42,
                    stratify=y if self.model_type == 'classification' else None
                )
            else:
                X_train, X_val, y_train, y_val = X, None, y, None
            
            # Préprocessing
            X_processed, y_processed = self.preprocess_data(X_train, fit_transform=True)
            
            # Construction du modèle si pas fait
            if self.model is None:
                self.model = self.build_model(**kwargs)
            
            # Entraînement
            start_time = datetime.now()
            
            if self.model_type == 'clustering':
                self.model.fit(X_processed)
            else:
                self.model.fit(X_processed, y_processed)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Validation si données disponibles
            validation_metrics = {}
            if X_val is not None and y_val is not None:
                validation_metrics = self.validate_model(X_val, y_val)
            
            # Métriques d'entraînement
            if self.model_type != 'clustering':
                train_predictions = self.model.predict(X_processed)
                train_metrics = self._calculate_metrics(y_processed, train_predictions)
            else:
                train_metrics = self._calculate_clustering_metrics(X_processed)
            
            # Sauvegarde historique
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'training_time': training_time,
                'train_samples': len(X_train),
                'validation_samples': len(X_val) if X_val is not None else 0,
                'train_metrics': train_metrics,
                'validation_metrics': validation_metrics,
                'config': kwargs
            }
            
            self.training_history.append(training_record)
            self.validation_scores = validation_metrics
            self.is_trained = True
            self.feature_names = list(X.columns)
            
            # Feature importance si disponible
            try:
                if hasattr(self.model, 'feature_importances_'):
                    self.feature_importance = dict(zip(
                        self.feature_names,
                        self.model.feature_importances_
                    ))
                elif hasattr(self.model, 'coef_'):
                    self.feature_importance = dict(zip(
                        self.feature_names,
                        abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else abs(self.model.coef_)
                    ))
            except:
                pass
            
            self.logger.info(f"✅ Entraînement terminé en {training_time:.2f}s")
            return training_record
            
        except Exception as e:
            self.logger.error(f"❌ Erreur entraînement: {e}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_proba: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        🔮 Prédictions sur nouvelles données
        """
        try:
            if not self.is_trained:
                raise ValueError("Modèle non entraîné")
            
            # Conversion et preprocessing
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.feature_names)
            
            X_processed, _ = self.preprocess_data(X, fit_transform=False)
            
            # Prédiction
            if return_proba and hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(X_processed)
            elif return_proba and hasattr(self.model, 'decision_function'):
                predictions = self.model.decision_function(X_processed)
            else:
                predictions = self.model.predict(X_processed)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Erreur prédiction: {e}")
            raise
    
    def validate_model(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """
        📊 Validation du modèle sur données test
        """
        try:
            X_val_processed, y_val_processed = self.preprocess_data(X_val, fit_transform=False)
            
            if self.model_type == 'clustering':
                return self._calculate_clustering_metrics(X_val_processed)
            
            predictions = self.model.predict(X_val_processed)
            metrics = self._calculate_metrics(y_val_processed, predictions)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Erreur validation: {e}")
            return {}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        📈 Calcul métriques selon type de modèle
        """
        metrics = {}
        
        if self.model_type == 'classification':
            metrics.update({
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            })
            
        elif self.model_type == 'regression':
            metrics.update({
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2_score': r2_score(y_true, y_pred)
            })
        
        return {k: round(v, 4) for k, v in metrics.items()}
    
    def _calculate_clustering_metrics(self, X: np.ndarray) -> Dict[str, float]:
        """
        📊 Métriques pour clustering
        """
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            labels = self.model.predict(X)
            
            metrics = {
                'silhouette_score': silhouette_score(X, labels),
                'calinski_harabasz_score': calinski_harabasz_score(X, labels),
                'n_clusters': len(np.unique(labels))
            }
            
            return {k: round(v, 4) for k, v in metrics.items()}
            
        except Exception as e:
            self.logger.warning(f"Impossible de calculer métriques clustering: {e}")
            return {}
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        💾 Sauvegarde du modèle complet
        """
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.models_dir / f"{self.model_name}_{timestamp}.pkl"
            
            # Sauvegarde modèle + métadonnées
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'preprocessing_pipeline': self.preprocessing_pipeline,
                'training_history': self.training_history,
                'validation_scores': self.validation_scores,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained,
                'config': self.config,
                'saved_at': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"💾 Modèle sauvé: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde: {e}")
            raise
    
    def load_model(self, filepath: str) -> bool:
        """
        📂 Chargement du modèle
        """
        try:
            model_data = joblib.load(filepath)
            
            # Restauration des attributs
            for key, value in model_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            self.logger.info(f"📂 Modèle chargé: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        📋 Informations complètes du modèle
        """
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'last_training': self.training_history[-1] if self.training_history else None,
            'validation_scores': self.validation_scores,
            'feature_importance': dict(sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )) if self.feature_importance else {},
            'config': self.config
        }
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """
        🔄 Validation croisée
        """
        try:
            if not self.is_trained:
                # Entraînement temporaire pour CV
                temp_model = self.build_model()
            else:
                temp_model = self.model
            
            X_processed, y_processed = self.preprocess_data(X, fit_transform=True)
            
            # Scores selon type
            if self.model_type == 'classification':
                scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            else:
                scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
            
            cv_results = {}
            for score in scoring:
                scores = cross_val_score(temp_model, X_processed, y_processed, cv=cv, scoring=score)
                cv_results[score] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores.tolist()
                }
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"❌ Erreur validation croisée: {e}")
            return {}

# ============================================================================
# 🧪 FONCTION TEST
# ============================================================================

def test_base_model():
    """Test de la classe BaseModel avec implémentation simple"""
    
    # Implémentation test
    class TestModel(BaseModel):
        def build_model(self, **kwargs):
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=10, random_state=42)
        
        def preprocess_data(self, data, fit_transform=True):
            # Preprocessing simple pour test
            from sklearn.preprocessing import LabelEncoder
            
            X = data.select_dtypes(include=[np.number]).fillna(0)
            y = None
            
            if 'target' in data.columns:
                y = data['target'].values
                X = X.drop('target', axis=1, errors='ignore')
            
            return X.values, y
    
    print("🧪 TEST BASE MODEL...")
    
    # Données test
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(2, 1.5, 1000),
        'feature3': np.random.uniform(0, 10, 1000),
        'target': np.random.choice([0, 1], 1000)
    })
    
    # Test modèle
    model = TestModel("test_model", "classification")
    
    # Validation données
    assert model.validate_data(data, ['feature1', 'feature2', 'target'])
    
    # Entraînement
    X = data[['feature1', 'feature2', 'feature3']]
    y = data['target']
    
    training_result = model.train(X, y, validation_split=0.2)
    assert model.is_trained
    assert 'train_metrics' in training_result
    
    # Prédictions
    predictions = model.predict(X.head(10))
    assert len(predictions) == 10
    
    # Sauvegarde/Chargement
    filepath = model.save_model()
    assert Path(filepath).exists()
    
    # Infos modèle
    info = model.get_model_info()
    assert info['is_trained']
    assert info['feature_count'] == 3
    
    print("✅ TOUS LES TESTS RÉUSSIS !")
    return True

# ============================================================================
# 🚀 EXECUTION
# ============================================================================

if __name__ == "__main__":
    test_base_model()
