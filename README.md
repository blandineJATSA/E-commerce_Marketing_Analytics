# E-commerce_Marketing_Analytics

# E-commerce MLOps Platform

## 🎯 Objectifs
Pipeline de Machine Learning pour l'analyse marketing e-commerce :
- Segmentation clients (RFM + Clustering)
- Prédiction de churn
- Optimisation de campagnes

## 🏗️ Architecture
- **Orchestration** : Apache Airflow
- **Tracking** : MLflow
- **API** : FastAPI
- **Monitoring** : Custom dashboards

## 🚀 Quick Start
1. Installation : `pip install -r requirements.txt`
2. Configuration : Modifier `config/config.yaml`
3. Données : Placer les fichiers dans `data/raw/`
4. Lancement : `python src/main.py`

## 📊 Données
- **Source** : Online Retail II Dataset
- **Format** : Excel (.xlsx)
- **Taille** : 1M+ transactions
- **Période** : 2009-2011

## 🔄 Pipeline

Data Raw → Preprocessing → Feature Engineering → Model Training → Deployment


## 📁 Structure
Voir `docs/architecture.md` pour détails complets.

⚙️ 2. Configuration Principale
# config/config.yaml
project:
  name: "e-commerce-mlops"
  version: "1.0.0"
  description: "MLOps Pipeline for E-commerce Marketing Analytics"

data:
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  features_path: "data/features/"
  file_name: "online_retail_II.xlsx"

mlflow:
  tracking_uri: "sqlite:///mlflow/mlflow.db"
  experiment_name: "e-commerce-segmentation"
  model_registry_uri: "sqlite:///mlflow/models.db"
  artifact_location: "mlflow/artifacts"

airflow:
  dags_folder: "airflow/dags"
  logs_folder: "airflow/logs"
  plugins_folder: "airflow/plugins"

models:
  segmentation:
    method: "kmeans"
    n_clusters: 5
    random_state: 42
  
  churn:
    algorithm: "random_forest"
    test_size: 0.2
    random_state: 42

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/pipeline.log"