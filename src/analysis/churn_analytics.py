"""
Churn Analytics - Analyse et prédiction du risque de désabonnement client
Scoring churn + Modèles ML + Actions préventives
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ChurnAnalytics:
    
    def __init__(self, config=None):
        """
        Analyseur de churn complet - Scoring + Prédiction + Actions
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
        # Paramètres par défaut
        self.churn_threshold_days = self.config.get('churn_threshold_days', 90)
        self.prediction_horizon_days = self.config.get('prediction_horizon_days', 30)
        
    def prepare_churn_features(self, orders_df: pd.DataFrame, 
                              customer_features: pd.DataFrame = None) -> pd.DataFrame:
        """
        📊 Prépare features pour modèle churn
        
        Args:
            orders_df: DataFrame commandes avec ['customer_id', 'order_date', 'total_amount']
            customer_features: Features additionnelles (optionnel)
            
        Returns:
            DataFrame avec features churn par client
        """
        try:
            self.logger.info("📊 Préparation features churn...")
            
            df = orders_df.copy()
            df['order_date'] = pd.to_datetime(df['order_date'])
            reference_date = df['order_date'].max()
            
            # ============================================================================
            # 🎯 FEATURES COMPORTEMENTALES CHURN
            # ============================================================================
            
            # Features de base par client
            customer_stats = df.groupby('customer_id').agg({
                'order_date': ['min', 'max', 'count'],
                'total_amount': ['sum', 'mean', 'std'],
                'customer_id': 'count'  # nombre commandes
            }).reset_index()
            
            # Flatten colonnes
            customer_stats.columns = ['customer_id', 'first_order_date', 'last_order_date', 
                                    'order_frequency', 'total_spent', 'avg_order_value', 
                                    'spending_std', 'total_orders']
            
            # ============================================================================
            # ⏰ FEATURES TEMPORELLES
            # ============================================================================
            
            # Recency (jours depuis dernière commande)
            customer_stats['recency_days'] = (reference_date - customer_stats['last_order_date']).dt.days
            
            # Customer lifetime (durée client en jours)
            customer_stats['customer_lifetime_days'] = (
                customer_stats['last_order_date'] - customer_stats['first_order_date']
            ).dt.days
            customer_stats['customer_lifetime_days'] = customer_stats['customer_lifetime_days'].fillna(0)
            
            # Fréquence d'achat (jours entre commandes)
            customer_stats['avg_days_between_orders'] = np.where(
                customer_stats['total_orders'] > 1,
                customer_stats['customer_lifetime_days'] / (customer_stats['total_orders'] - 1),
                999  # Nouveaux clients = valeur élevée
            )
            
            # ============================================================================
            # 📈 FEATURES TENDANCES (3 derniers mois vs historique)
            # ============================================================================
            
            # Période récente (90 derniers jours)
            recent_cutoff = reference_date - timedelta(days=90)
            recent_orders = df[df['order_date'] >= recent_cutoff]
            
            recent_stats = recent_orders.groupby('customer_id').agg({
                'total_amount': ['sum', 'mean', 'count'],
                'order_date': 'count'
            }).reset_index()
            
            recent_stats.columns = ['customer_id', 'recent_spent', 'recent_aov', 
                                  'recent_orders', 'recent_orders_count']
            
            # Merge avec stats globales
            customer_stats = customer_stats.merge(recent_stats, on='customer_id', how='left')
            customer_stats = customer_stats.fillna(0)
            
            # Ratios d'évolution (récent vs historique)
            customer_stats['spending_trend'] = np.where(
                customer_stats['total_spent'] > 0,
                customer_stats['recent_spent'] / (customer_stats['total_spent'] / 
                (customer_stats['customer_lifetime_days'] + 1) * 90),  # Pro-rata 90j
                0
            )
            
            customer_stats['order_frequency_trend'] = np.where(
                customer_stats['total_orders'] > 0,
                customer_stats['recent_orders'] / (customer_stats['total_orders'] / 
                (customer_stats['customer_lifetime_days'] + 1) * 90),
                0
            )
            
            # ============================================================================
            # 🎯 FEATURES RISQUE
            # ============================================================================
            
            # Volatilité des achats (coefficient de variation)
            customer_stats['spending_volatility'] = np.where(
                customer_stats['avg_order_value'] > 0,
                customer_stats['spending_std'] / customer_stats['avg_order_value'],
                0
            )
            
            # Score d'engagement (commandes par mois d'activité)
            customer_stats['engagement_score'] = np.where(
                customer_stats['customer_lifetime_days'] > 0,
                customer_stats['total_orders'] / (customer_stats['customer_lifetime_days'] / 30.44),
                customer_stats['total_orders']  # Nouveaux clients
            )
            
            # Ratio première/dernière commande (détection one-shot)
            customer_stats['is_one_time_customer'] = (customer_stats['total_orders'] == 1).astype(int)
            
            # ============================================================================
            # 🔄 MERGE AVEC FEATURES EXTERNES
            # ============================================================================
            
            if customer_features is not None:
                customer_stats = customer_stats.merge(
                    customer_features[['customer_id', 'product_diversity', 'clv_estimated']], 
                    on='customer_id', how='left'
                )
                customer_stats = customer_stats.fillna(0)
            
            # Nettoyage final
            customer_stats = customer_stats.fillna(0)
            customer_stats = customer_stats.replace([np.inf, -np.inf], 0)
            
            self.logger.info(f"✅ Features churn préparées pour {len(customer_stats)} clients")
            return customer_stats
            
        except Exception as e:
            self.logger.error(f"❌ Erreur préparation features churn: {e}")
            raise
    
    def define_churn_labels(self, churn_features: pd.DataFrame) -> pd.DataFrame:
        """
        🎯 Définit les labels churn (0=Actif, 1=Churned)
        
        Args:
            churn_features: DataFrame avec features clients
            
        Returns:
            DataFrame avec colonne 'is_churned'
        """
        try:
            self.logger.info(f"🎯 Définition labels churn (seuil: {self.churn_threshold_days} jours)...")
            
            df = churn_features.copy()
            
            # ============================================================================
            # 📏 RÈGLES DÉFINITION CHURN
            # ============================================================================
            
            # Règle 1: Recency > seuil ET client pas nouveau
            basic_churn = (
                (df['recency_days'] > self.churn_threshold_days) & 
                (df['customer_lifetime_days'] > 30)  # Pas les tout nouveaux clients
            )
            
            # Règle 2: One-time customer + recency élevée
            onetime_churn = (
                (df['is_one_time_customer'] == 1) & 
                (df['recency_days'] > 60)  # Seuil plus bas pour one-time
            )
            
            # Règle 3: Forte baisse d'engagement récent
            engagement_churn = (
                (df['spending_trend'] < 0.3) &  # Baisse 70%+ des achats récents
                (df['order_frequency_trend'] < 0.3) &
                (df['total_orders'] > 2)  # Client établi
            )
            
            # Label final
            df['is_churned'] = (basic_churn | onetime_churn | engagement_churn).astype(int)
            
            # Stats
            churn_rate = df['is_churned'].mean() * 100
            churned_count = df['is_churned'].sum()
            active_count = len(df) - churned_count
            
            self.logger.info(f"✅ Labels créés: {active_count} actifs, {churned_count} churned ({churn_rate:.1f}% churn)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erreur définition labels: {e}")
            raise
    
    def train_churn_models(self, labeled_data: pd.DataFrame) -> Dict:
        """
        🤖 Entraîne modèles ML de prédiction churn
        """
        try:
            self.logger.info("🤖 Entraînement modèles churn...")
            
            # Features pour ML
            ml_features = [
                'recency_days', 'total_orders', 'avg_order_value', 'total_spent',
                'customer_lifetime_days', 'avg_days_between_orders',
                'recent_spent', 'recent_orders', 'spending_trend', 'order_frequency_trend',
                'spending_volatility', 'engagement_score', 'is_one_time_customer'
            ]
            
            # Ajouter features externes si disponibles
            optional_features = ['product_diversity', 'clv_estimated']
            for feat in optional_features:
                if feat in labeled_data.columns:
                    ml_features.append(feat)
            
            # Préparation données
            X = labeled_data[ml_features].copy()
            y = labeled_data['is_churned'].copy()
            
            # Nettoyage
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Standardisation
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # ============================================================================
            # 🎯 ENTRAÎNEMENT MULTIPLE MODÈLES
            # ============================================================================
            
            models_config = {
                'logistic': LogisticRegression(random_state=42, class_weight='balanced'),
                'random_forest': RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight='balanced'
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100, random_state=42
                )
            }
            
            model_results = {}
            
            for name, model in models_config.items():
                self.logger.info(f"   Entraînement {name}...")
                
                if name == 'logistic':
                    # Régression logistique avec features standardisées
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    # Tree-based models avec features originales
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
                
                # Métriques
                auc_score = roc_auc_score(y_test, y_proba)
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                # Feature importance (pour tree-based)
                if hasattr(model, 'feature_importances_'):
                    feature_imp = pd.DataFrame({
                        'feature': ml_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                else:
                    feature_imp = pd.DataFrame({
                        'feature': ml_features,
                        'importance': abs(model.coef_[0])
                    }).sort_values('importance', ascending=False)
                
                model_results[name] = {
                    'model': model,
                    'auc_score': auc_score,
                    'confusion_matrix': conf_matrix,
                    'classification_report': class_report,
                    'feature_importance': feature_imp,
                    'test_predictions': y_proba
                }
                
                self.logger.info(f"     → AUC: {auc_score:.3f}")
            
            # Sélection meilleur modèle
            best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['auc_score'])
            best_model = model_results[best_model_name]
            
            self.models = model_results
            self.feature_importance = best_model['feature_importance']
            
            training_results = {
                'models': model_results,
                'best_model': best_model_name,
                'best_auc': best_model['auc_score'],
                'ml_features': ml_features,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"✅ Modèles entraînés - Meilleur: {best_model_name} (AUC: {best_model['auc_score']:.3f})")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ Erreur entraînement modèles: {e}")
            raise
    
    def predict_churn_risk(self, customer_data: pd.DataFrame, 
                          model_name: str = 'random_forest') -> pd.DataFrame:
        """
        🎯 Prédit risque churn pour nouveaux clients
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Modèle {model_name} non trouvé. Modèles disponibles: {list(self.models.keys())}")
            
            model = self.models[model_name]['model']
            ml_features = self.models[model_name].get('ml_features', [])
            
            # Préparer features
            X = customer_data[ml_features].copy()
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Prédiction
            if model_name == 'logistic':
                X_scaled = self.scaler.transform(X)
                churn_probability = model.predict_proba(X_scaled)[:, 1]
            else:
                churn_probability = model.predict_proba(X)[:, 1]
            
            # Scoring risque
            def risk_category(prob):
                if prob >= 0.8:
                    return 'Critical'
                elif prob >= 0.6:
                    return 'High' 
                elif prob >= 0.4:
                    return 'Medium'
                elif prob >= 0.2:
                    return 'Low'
                else:
                    return 'Very Low'
            
            # Résultats
            results = customer_data[['customer_id']].copy()
            results['churn_probability'] = churn_probability
            results['risk_category'] = results['churn_probability'].apply(risk_category)
            results['model_used'] = model_name
            results['prediction_date'] = datetime.now().strftime('%Y-%m-%d')
            
            self.logger.info(f"✅ Prédictions churn générées pour {len(results)} clients")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Erreur prédiction churn: {e}")
            raise
    
    def generate_retention_actions(self, churn_predictions: pd.DataFrame,
                                 customer_segments: pd.DataFrame = None) -> pd.DataFrame:
        """
        💡 Génère actions de rétention personnalisées
        """
        try:
            self.logger.info("💡 Génération actions rétention...")
            
            df = churn_predictions.copy()
            
            # Merge avec segments si disponible
            if customer_segments is not None:
                df = df.merge(
                    customer_segments[['customer_id', 'rfm_segment', 'monetary']], 
                    on='customer_id', how='left'
                )
            
            # ============================================================================
            # 🎯 MATRICE ACTIONS PAR RISQUE × SEGMENT
            # ============================================================================
            
            def get_retention_strategy(row):
                risk = row['risk_category']
                segment = row.get('rfm_segment', 'Unknown')
                value = row.get('monetary', 0)
                
                # Stratégies par risque
                if risk == 'Critical':
                    if value > 1000:  # High-value
                        return {
                            'priority': 'P0 - Immediate',
                            'channel': 'Personal call + Email',
                            'offer': 'Exclusive 30% discount + VIP benefits',
                            'budget': 'High ($100+)',
                            'timeline': '24-48h',
                            'actions': ['Personal outreach', 'Custom offer', 'Account manager contact']
                        }
                    else:
                        return {
                            'priority': 'P1 - Urgent',
                            'channel': 'Email + SMS',
                            'offer': '25% discount + Free shipping',
                            'budget': 'Medium ($50)',
                            'timeline': '3-7 days',
                            'actions': ['Win-back email series', 'Limited time offer', 'Product recommendations']
                        }
                
                elif risk == 'High':
                    if segment == 'Champions':
                        return {
                            'priority': 'P1 - Urgent',
                            'channel': 'Email + Phone',
                            'offer': 'VIP early access + 20% off',
                            'budget': 'High ($75)',
                            'timeline': '1 week',
                            'actions': ['VIP re-engagement', 'Loyalty program promotion', 'New arrivals preview']
                        }
                    else:
                        return {
                            'priority': 'P2 - Important',
                            'channel': 'Email sequence',
                            'offer': '20% discount on favorites',
                            'budget': 'Medium ($30)',
                            'timeline': '2 weeks',
                            'actions': ['Personalized recommendations', 'Browsing behavior triggers', 'Social proof']
                        }
                
                elif risk == 'Medium':
                    return {
                        'priority': 'P3 - Monitor',
                        'channel': 'Automated email',
                        'offer': '15% off next purchase',
                        'budget': 'Low ($20)',
                        'timeline': '1 month',
                        'actions': ['Newsletter engagement', 'Product education', 'Community building']
                    }
                
                else:  # Low/Very Low
                    return {
                        'priority': 'P4 - Maintain',
                        'channel': 'Regular marketing',
                        'offer': 'Standard promotions',
                        'budget': 'Standard ($10)',
                        'timeline': 'Ongoing',
                        'actions': ['Regular newsletter', 'Seasonal campaigns', 'Cross-selling']
                    }
            
            # Appliquer stratégies
            strategies = df.apply(get_retention_strategy, axis=1)
            
            # Extraire colonnes
            for key in ['priority', 'channel', 'offer', 'budget', 'timeline']:
                df[f'retention_{key}'] = strategies.apply(lambda x: x[key])
            
            df['retention_actions'] = strategies.apply(lambda x: ', '.join(x['actions']))
            
            # ROI estimé par action
            budget_mapping = {'High ($100+)': 100, 'High ($75)': 75, 'Medium ($50)': 50, 
                            'Medium ($30)': 30, 'Low ($20)': 20, 'Standard ($10)': 10}
            
            df['estimated_budget'] = df['retention_budget'].map(budget_mapping)
            df['estimated_success_rate'] = df['risk_category'].map({
                'Critical': 0.15, 'High': 0.25, 'Medium': 0.35, 'Low': 0.45, 'Very Low': 0.60
            })
            
            # ROI simple (si client revient avec 1 commande moyenne)
            avg_order_value = customer_segments['monetary'].mean() if customer_segments is not None else 100
            df['estimated_roi'] = (
                (df['estimated_success_rate'] * avg_order_value * 0.2 - df['estimated_budget']) / 
                df['estimated_budget'] * 100
            ).round(1)
            
            self.logger.info("✅ Actions rétention générées")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erreur génération actions: {e}")
            raise
    
    def analyze_churn_patterns(self, labeled_data: pd.DataFrame) -> Dict:
        """
        📊 Analyse patterns et insights churn
        """
        try:
            self.logger.info("📊 Analyse patterns churn...")
            
            churned = labeled_data[labeled_data['is_churned'] == 1]
            active = labeled_data[labeled_data['is_churned'] == 0]
            
            patterns = {
                'churn_rate': (len(churned) / len(labeled_data) * 100).round(1),
                'churned_customers': len(churned),
                'active_customers': len(active),
                
                # Comparaisons moyennes
                'avg_recency_churned': churned['recency_days'].mean().round(1),
                'avg_recency_active': active['recency_days'].mean().round(1),
                
                'avg_orders_churned': churned['total_orders'].mean().round(1),
                'avg_orders_active': active['total_orders'].mean().round(1),
                
                'avg_value_churned': churned['total_spent'].mean().round(2),
                'avg_value_active': active['total_spent'].mean().round(2),
                
                # Top risk factors
                'one_time_customer_churn_rate': (
                    churned['is_one_time_customer'].sum() / 
                    labeled_data['is_one_time_customer'].sum() * 100
                ).round(1) if labeled_data['is_one_time_customer'].sum() > 0 else 0,
                
                'revenue_at_risk': churned['total_spent'].sum().round(2),
                
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"❌ Erreur analyse patterns: {e}")
            return {'error': str(e)}
    
    def run_complete_churn_analysis(self, orders_df: pd.DataFrame,
                                  customer_segments: pd.DataFrame = None) -> Dict:
        """
        🚀 Pipeline complet d'analyse churn
        """
        try:
            self.logger.info("🚀 Début analyse churn complète...")
            
            # 1. Préparation features
            churn_features = self.prepare_churn_features(orders_df, customer_segments)
            
            # 2. Labels churn
            labeled_data = self.define_churn_labels(churn_features)
            
            # 3. Entraînement modèles
            training_results = self.train_churn_models(labeled_data)
            
            # 4. Prédictions sur tous les clients
            predictions = self.predict_churn_risk(
                labeled_data, 
                model_name=training_results['best_model']
            )
            
            # 5. Actions de rétention
            retention_actions = self.generate_retention_actions(
                predictions, customer_segments
            )
            
            # 6. Analyse patterns
            churn_patterns = self.analyze_churn_patterns(labeled_data)
            
            # Résultat complet
            complete_result = {
                'churn_predictions': retention_actions,
                'model_performance': training_results,
                'churn_patterns': churn_patterns,
                'feature_importance': self.feature_importance,
                'models': self.models,
                'analysis_config': {
                    'churn_threshold_days': self.churn_threshold_days,
                    'prediction_date': datetime.now().strftime('%Y-%m-%d')
                }
            }
            
            self.logger.info("🎉 Analyse churn complète terminée !")
            return complete_result
            
        except Exception as e:
            self.logger.error(f"❌ Erreur analyse churn complète: {e}")
            raise

# ============================================================================
# 🧪 FONCTION TEST
# ============================================================================

def test_churn_analytics():
    """
    Test avec données simulées
    """
    print("🧪 TEST CHURN ANALYTICS")
    print("=" * 50)
    
    # Données simulées (clients avec patterns churn réalistes)
    np.random.seed(42)
    n_customers = 1000
    
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    orders_data = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    for customer_id in customer_ids:
        # Simulation différents patterns
        
        if np.random.random() < 0.7:  # 70% clients normaux
            n_orders = np.random.poisson(3) + 1
            for i in range(n_orders):
                # Dernières commandes plus récentes pour clients actifs
                if np.random.random() < 0.8:  # 80% clients actifs
                    days_ago = np.random.randint(0, 60)  # Commandes récentes
                else:  # 20% churned (commandes anciennes)
                    days_ago = np.random.randint(120, 300)
                    
                order_date = end_date - timedelta(days=days_ago)
                total_amount = np.random.uniform(50, 300)
                
                orders_data.append({
                    'customer_id': customer_id,
                    'order_date': order_date,
                    'total_amount': round(total_amount, 2)
                })
        
        else:  # 30% one-time customers (plus prone to churn)
            days_ago = np.random.randint(30, 200)  # Une seule commande
            order_date = end_date - timedelta(days=days_ago)
            total_amount = np.random.uniform(20, 150)
            
            orders_data.append({
                'customer_id': customer_id,
                'order_date': order_date,
                'total_amount': round(total_amount, 2)
            })
    
    orders_df = pd.DataFrame(orders_data)
    
    # Test analyse
    analyzer = ChurnAnalytics()
    
    try:
        result = analyzer.run_complete_churn_analysis(orders_df)
        
        print("\n🎯 RÉSULTATS TEST :")
        print(f"✅ Clients analysés: {len(result['churn_predictions'])}")
        print(f"✅ Taux de churn: {result['churn_patterns']['churn_rate']}%")
        print(f"✅ Meilleur modèle: {result['model_performance']['best_model']}")
        print(f"✅ AUC score: {result['model_performance']['best_auc']:.3f}")
        
        # Distribution risques
        print("\n📊 DISTRIBUTION RISQUES :")
        risk_dist = result['churn_predictions']['risk_category'].value_counts()
        for risk, count in risk_dist.items():
            print(f"   {risk}: {count} clients ({count/len(result['churn_predictions'])*100:.1f}%)")
        
        # Top features importantes
        print("\n🎯 TOP 5 FEATURES IMPORTANTES :")
        top_features = result['feature_importance'].head()
        for _, row in top_features.iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        print("\n🎉 TEST RÉUSSI !")
        return True
        
    except Exception as e:
        print(f"❌ ERREUR TEST: {e}")
        return False

# ============================================================================
# 🚀 EXECUTION TEST
# ============================================================================

if __name__ == "__main__":
    test_churn_analytics()
