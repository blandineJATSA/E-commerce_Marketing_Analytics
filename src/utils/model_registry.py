# src/utils/mlflow_tracker.py
import mlflow
import mlflow.sklearn
import yaml
from pathlib import Path
import logging

class MLflowTracker:
    def __init__(self, config_path="config/mlflow_config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_mlflow()
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path):
        """Charge la configuration MLflow"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _setup_mlflow(self):
        """Configure MLflow"""
        mlflow_config = self.config['mlflow']
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        
        # Créer les dossiers nécessaires
        Path("mlflow/tracking").mkdir(parents=True, exist_ok=True)
        Path("mlflow/models").mkdir(parents=True, exist_ok=True)
        Path("mlflow/experiments").mkdir(parents=True, exist_ok=True)
    
    def start_experiment(self, experiment_name):
        """Démarre une expérience MLflow"""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            self.logger.info(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            self.logger.info(f"Using existing experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
    
    def start_run(self, run_name=None, tags=None):
        """Démarre un run MLflow"""
        return mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_params(self, params):
        """Log les paramètres"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics):
        """Log les métriques"""
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, artifact_path, **kwargs):
        """Log un modèle"""
        mlflow.sklearn.log_model(model, artifact_path, **kwargs)
    
    def log_artifact(self, local_path, artifact_path=None):
        """Log un artifact"""
        mlflow.log_artifact(local_path, artifact_path)
