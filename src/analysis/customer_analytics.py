"""
Customer Analytics - Analyse et segmentation client complète
Reproduction des approches RFM + K-means + CLV
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class CustomerAnalytics:
    
    def __init__(self, config=None):
        """
        Analyseur client complet - RFM + Clustering + CLV
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.pca_model = None
        
    def calculate_rfm(self, orders_df: pd.DataFrame, reference_date: str = None) -> pd.DataFrame:
        """
        🎯 Calcule les scores RFM (Recency, Frequency, Monetary)
        
        Args:
            orders_df: DataFrame avec ['customer_id', 'order_date', 'total_amount']
            reference_date: Date de référence (défaut: date max)
            
        Returns:
            DataFrame avec scores RFM par client
        """
        try:
            self.logger.info("🎯 Calcul RFM en cours...")
            
            # Préparation données
            df = orders_df.copy()
            df['order_date'] = pd.to_datetime(df['order_date'])
            
            if reference_date is None:
                reference_date = df['order_date'].max()
            else:
                reference_date = pd.to_datetime(reference_date)
            
            # Calcul métriques RFM
            rfm = df.groupby('customer_id').agg({
                'order_date': lambda x: (reference_date - x.max()).days,  # Recency
                'customer_id': 'count',  # Frequency  
                'total_amount': ['sum', 'mean']  # Monetary + AOV
            }).reset_index()
            
            # Flatten colonnes
            rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary', 'avg_order_value']
            
            # Nettoyage
            rfm = rfm[rfm['monetary'] > 0].copy()
            
            # 📊 SCORING RFM (1-5) avec quintiles
            rfm['R_score'] = pd.qcut(rfm['recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
            rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
            rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
            
            # Scores numériques
            rfm['R_score'] = rfm['R_score'].astype(int)
            rfm['F_score'] = rfm['F_score'].astype(int)
            rfm['M_score'] = rfm['M_score'].astype(int)
            
            # Score total
            rfm['rfm_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
            
            self.logger.info(f"✅ RFM calculé pour {len(rfm)} clients")
            return rfm
            
        except Exception as e:
            self.logger.error(f"❌ Erreur calcul RFM: {e}")
            raise
    
    def segment_customers_rfm(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        🎯 Segmentation clients basée sur RFM (version simple)
        """
        try:
            df = rfm_df.copy()
            
            def categorize_rfm(score):
                if score >= 13:
                    return 'Champions'
                elif score >= 10:
                    return 'Loyal_Customers'
                elif score >= 7:
                    return 'Potential_Loyalists'
                elif score >= 5:
                    return 'At_Risk'
                else:
                    return 'Lost_Customers'
            
            df['rfm_segment'] = df['rfm_score'].apply(categorize_rfm)
            
            self.logger.info("✅ Segmentation RFM terminée")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erreur segmentation RFM: {e}")
            raise
    
    def calculate_customer_features(self, orders_df: pd.DataFrame, 
                                  products_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        📊 Calcule features comportementales avancées pour clustering
        """
        try:
            self.logger.info("📊 Calcul features comportementales...")
            
            df = orders_df.copy()
            df['order_date'] = pd.to_datetime(df['order_date'])
            
            # Features par client
            customer_features = df.groupby('customer_id').agg({
                'order_date': ['min', 'max', 'count'],
                'total_amount': ['sum', 'mean', 'std'],
                'quantity': ['sum', 'mean'] if 'quantity' in df.columns else ['count', 'count']
            }).reset_index()
            
            # Flatten colonnes
            customer_features.columns = ['customer_id', 'first_order', 'last_order', 
                                       'frequency', 'monetary', 'avg_order_value', 
                                       'monetary_std', 'total_quantity', 'avg_quantity']
            
            # Features temporelles
            reference_date = df['order_date'].max()
            customer_features['recency'] = (reference_date - customer_features['last_order']).dt.days
            customer_features['customer_lifetime'] = (customer_features['last_order'] - customer_features['first_order']).dt.days
            customer_features['customer_lifetime'] = customer_features['customer_lifetime'].fillna(0)
            
            # Features behaviorales
            customer_features['avg_days_between_orders'] = np.where(
                customer_features['frequency'] > 1,
                customer_features['customer_lifetime'] / (customer_features['frequency'] - 1),
                0
            )
            
            # Diversité produits (si disponible)
            if products_df is not None and 'product_id' in df.columns:
                product_diversity = df.groupby('customer_id')['product_id'].nunique().reset_index()
                product_diversity.columns = ['customer_id', 'product_diversity']
                customer_features = customer_features.merge(product_diversity, on='customer_id', how='left')
            else:
                customer_features['product_diversity'] = 1
            
            # Nettoyage
            customer_features = customer_features.fillna(0)
            customer_features = customer_features[customer_features['monetary'] > 0]
            
            self.logger.info(f"✅ Features calculées pour {len(customer_features)} clients")
            return customer_features
            
        except Exception as e:
            self.logger.error(f"❌ Erreur calcul features: {e}")
            raise
    
    def perform_clustering(self, customer_features: pd.DataFrame, 
                         n_clusters: int = 5) -> pd.DataFrame:
        """
        🤖 Clustering K-means comportemental
        """
        try:
            self.logger.info(f"🤖 Clustering K-means ({n_clusters} clusters)...")
            
            # Sélection features pour clustering
            clustering_features = [
                'recency', 'frequency', 'monetary', 'avg_order_value',
                'customer_lifetime', 'avg_days_between_orders', 'product_diversity'
            ]
            
            # Préparer données
            cluster_data = customer_features[['customer_id'] + clustering_features].copy()
            cluster_data = cluster_data.dropna()
            
            # Standardisation
            X = self.scaler.fit_transform(cluster_data[clustering_features])
            
            # K-means
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.kmeans_model.fit_predict(X)
            
            # PCA pour visualisation
            self.pca_model = PCA(n_components=2, random_state=42)
            pca_result = self.pca_model.fit_transform(X)
            
            # Ajouter résultats
            cluster_data['cluster'] = cluster_labels
            cluster_data['pca_1'] = pca_result[:, 0]
            cluster_data['pca_2'] = pca_result[:, 1]
            
            # Merge avec features originales
            result = customer_features.merge(
                cluster_data[['customer_id', 'cluster', 'pca_1', 'pca_2']], 
                on='customer_id', how='left'
            )
            
            self.logger.info(f"✅ Clustering terminé - {n_clusters} clusters")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Erreur clustering: {e}")
            raise
    
    def calculate_clv(self, customer_data: pd.DataFrame, 
                     prediction_months: int = 12) -> pd.DataFrame:
        """
        💰 Calcule Customer Lifetime Value estimé
        """
        try:
            self.logger.info("💰 Calcul CLV en cours...")
            
            df = customer_data.copy()
            
            # CLV simple basé sur fréquence et valeur moyenne
            # CLV = (AOV × Fréquence mensuelle × Durée de vie estimée) × Marge
            
            # Fréquence mensuelle estimée
            monthly_frequency = np.where(
                df['customer_lifetime'] > 0,
                df['frequency'] / (df['customer_lifetime'] / 30.44),  # 30.44 = jours moyen/mois
                df['frequency'] / prediction_months  # Si nouveau client
            )
            
            # Durée de vie estimée (en mois)
            estimated_lifetime = np.where(
                df['frequency'] > 1,
                prediction_months,  # Clients récurrents: prédiction complète
                prediction_months * 0.3  # Nouveaux clients: durée réduite
            )
            
            # CLV (assumons marge de 20%)
            gross_margin = 0.20
            df['clv_estimated'] = (
                df['avg_order_value'] * 
                monthly_frequency * 
                estimated_lifetime * 
                gross_margin
            )
            
            # CLV historique
            df['clv_historical'] = df['monetary'] * gross_margin
            
            self.logger.info("✅ CLV calculé")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erreur calcul CLV: {e}")
            raise
    
    def analyze_segments(self, segmented_data: pd.DataFrame) -> Dict:
        """
        📊 Analyse détaillée des segments
        """
        try:
            self.logger.info("📊 Analyse segments en cours...")
            
            # Analyse RFM segments
            rfm_analysis = segmented_data.groupby('rfm_segment').agg({
                'customer_id': 'count',
                'monetary': ['mean', 'sum'],
                'frequency': 'mean',
                'recency': 'mean',
                'avg_order_value': 'mean',
                'clv_estimated': ['mean', 'sum'] if 'clv_estimated' in segmented_data.columns else ['count', 'count']
            }).round(2)
            
            # Analyse clusters
            cluster_analysis = segmented_data.groupby('cluster').agg({
                'customer_id': 'count',
                'monetary': ['mean', 'sum'],
                'frequency': 'mean',
                'recency': 'mean',
                'avg_order_value': 'mean',
                'product_diversity': 'mean',
                'clv_estimated': ['mean', 'sum'] if 'clv_estimated' in segmented_data.columns else ['count', 'count']
            }).round(2)
            
            # Insights automatiques
            insights = self._generate_insights(segmented_data)
            
            analysis_result = {
                'rfm_segments': rfm_analysis,
                'clusters': cluster_analysis,
                'insights': insights,
                'total_customers': len(segmented_data),
                'total_revenue': segmented_data['monetary'].sum(),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info("✅ Analyse segments terminée")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ Erreur analyse segments: {e}")
            raise
    
    def _generate_insights(self, data: pd.DataFrame) -> Dict:
        """
        💡 Génère insights automatiques
        """
        insights = {}
        
        try:
            # Top segments par valeur
            if 'rfm_segment' in data.columns:
                top_rfm_segment = data.groupby('rfm_segment')['monetary'].sum().idxmax()
                insights['top_rfm_segment'] = top_rfm_segment
            
            # Top cluster par valeur
            if 'cluster' in data.columns:
                top_cluster = data.groupby('cluster')['monetary'].mean().idxmax()
                insights['highest_value_cluster'] = int(top_cluster)
            
            # Statistiques globales
            insights['top_20_percent_revenue_share'] = (
                data.nlargest(int(len(data) * 0.2), 'monetary')['monetary'].sum() / 
                data['monetary'].sum() * 100
            ).round(1)
            
            # Clients one-shot
            one_shot_clients = len(data[data['frequency'] == 1])
            insights['one_shot_clients_pct'] = (one_shot_clients / len(data) * 100).round(1)
            
        except Exception as e:
            insights['error'] = str(e)
        
        return insights
    
    def get_segment_recommendations(self, analysis_result: Dict) -> Dict:
        """
        🎯 Recommandations marketing par segment
        """
        recommendations = {
            'Champions': {
                'strategy': 'Programme VIP exclusif',
                'actions': ['Offres personnalisées', 'Early access produits', 'Support premium'],
                'budget_priority': 'High',
                'expected_roi': 'Very High'
            },
            'Loyal_Customers': {
                'strategy': 'Fidélisation et upselling',
                'actions': ['Cross-selling', 'Programme fidélité', 'Contenus exclusifs'],
                'budget_priority': 'High', 
                'expected_roi': 'High'
            },
            'Potential_Loyalists': {
                'strategy': 'Conversion en clients fidèles',
                'actions': ['Onboarding séquentiel', 'Offres incitatives', 'Social proof'],
                'budget_priority': 'Medium',
                'expected_roi': 'Medium'
            },
            'At_Risk': {
                'strategy': 'Réactivation urgente',
                'actions': ['Win-back campaigns', 'Offres spéciales', 'Feedback survey'],
                'budget_priority': 'Medium',
                'expected_roi': 'Medium'
            },
            'Lost_Customers': {
                'strategy': 'Reconquête ciblée',
                'actions': ['Campagnes réactivation', 'Deep discounts', 'Nouveautés'],
                'budget_priority': 'Low',
                'expected_roi': 'Low'
            }
        }
        
        return recommendations
    
    def run_complete_analysis(self, orders_df: pd.DataFrame, 
                            products_df: pd.DataFrame = None) -> Dict:
        """
        🚀 Analyse complète client - Pipeline principal
        """
        try:
            self.logger.info("🚀 Début analyse complète client...")
            
            # 1. Calcul RFM
            rfm_data = self.calculate_rfm(orders_df)
            
            # 2. Segmentation RFM
            rfm_segmented = self.segment_customers_rfm(rfm_data)
            
            # 3. Features comportementales
            customer_features = self.calculate_customer_features(orders_df, products_df)
            
            # 4. Clustering
            clustered_data = self.perform_clustering(customer_features)
            
            # 5. CLV
            clv_data = self.calculate_clv(clustered_data)
            
            # 6. Merge RFM + Clustering + CLV
            final_data = rfm_segmented[['customer_id', 'rfm_segment', 'rfm_score', 'R_score', 'F_score', 'M_score']].merge(
                clv_data, on='customer_id', how='inner'
            )
            
            # 7. Analyse segments
            analysis_result = self.analyze_segments(final_data)
            
            # 8. Recommandations
            recommendations = self.get_segment_recommendations(analysis_result)
            
            # Résultat complet
            complete_result = {
                'segmented_customers': final_data,
                'analysis': analysis_result,
                'recommendations': recommendations,
                'models': {
                    'kmeans': self.kmeans_model,
                    'scaler': self.scaler,
                    'pca': self.pca_model
                }
            }
            
            self.logger.info("🎉 Analyse complète terminée avec succès !")
            return complete_result
            
        except Exception as e:
            self.logger.error(f"❌ Erreur analyse complète: {e}")
            raise

# ============================================================================
# 🧪 FONCTION TEST
# ============================================================================

def test_customer_analytics():
    """
    Test avec données simulées
    """
    print("🧪 TEST CUSTOMER ANALYTICS")
    print("=" * 50)
    
    # Données simulées
    np.random.seed(42)
    n_customers = 1000
    n_orders = 3000
    
    # Génération clients
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    
    # Génération commandes
    orders_data = []
    start_date = datetime(2023, 1, 1)
    
    for _ in range(n_orders):
        customer_id = np.random.choice(customer_ids)
        order_date = start_date + timedelta(days=np.random.randint(0, 365))
        
        # Simulation différents profils clients
        if np.random.random() < 0.1:  # 10% champions
            total_amount = np.random.uniform(500, 2000)
        elif np.random.random() < 0.3:  # 30% loyal
            total_amount = np.random.uniform(100, 500)
        else:  # 60% autres
            total_amount = np.random.uniform(20, 200)
            
        orders_data.append({
            'customer_id': customer_id,
            'order_date': order_date,
            'total_amount': round(total_amount, 2),
            'quantity': np.random.randint(1, 10)
        })
    
    orders_df = pd.DataFrame(orders_data)
    
    # Test analyse
    analyzer = CustomerAnalytics()
    
    try:
        result = analyzer.run_complete_analysis(orders_df)
        
        print("\n🎯 RÉSULTATS TEST :")
        print(f"✅ Clients analysés: {len(result['segmented_customers'])}")
        print(f"✅ Segments RFM: {result['segmented_customers']['rfm_segment'].nunique()}")
        print(f"✅ Clusters: {result['segmented_customers']['cluster'].nunique()}")
        print(f"✅ Revenue total: ${result['analysis']['total_revenue']:,.2f}")
        
        # Distribution segments
        print("\n📊 DISTRIBUTION SEGMENTS :")
        segment_dist = result['segmented_customers']['rfm_segment'].value_counts()
        for segment, count in segment_dist.items():
            print(f"   {segment}: {count} clients ({count/len(result['segmented_customers'])*100:.1f}%)")
        
        print("\n🎉 TEST RÉUSSI !")
        return True
        
    except Exception as e:
        print(f"❌ ERREUR TEST: {e}")
        return False

# ============================================================================
# 🚀 EXECUTION TEST
# ============================================================================

if __name__ == "__main__":
    test_customer_analytics()
