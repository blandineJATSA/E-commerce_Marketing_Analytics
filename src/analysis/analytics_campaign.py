"""
Campaign Analytics - Optimisation campagnes marketing et ROI
Ciblage segments + A/B Testing + Attribution + Budget optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class CampaignAnalytics:
    
    def __init__(self, config=None):
        """
        Analyseur de campagnes marketing - ROI + Ciblage + Attribution
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.campaign_models = {}
        
        # Configuration par défaut
        self.default_costs = {
            'email': 0.10,
            'sms': 0.20, 
            'social_media': 2.00,
            'google_ads': 5.00,
            'display_ads': 3.00,
            'personal_call': 25.00,
            'direct_mail': 8.00
        }
        self.profit_margin = self.config.get('profit_margin', 0.25)  # 25% default
        
    def create_campaign_segments(self, customer_segments: pd.DataFrame,
                               churn_predictions: pd.DataFrame = None) -> pd.DataFrame:
        """
        🎯 Crée segments optimisés pour campagnes marketing
        
        Args:
            customer_segments: Segments RFM/clustering existants
            churn_predictions: Prédictions churn (optionnel)
            
        Returns:
            DataFrame avec segments campagne enrichis
        """
        try:
            self.logger.info("🎯 Création segments campagnes...")
            
            df = customer_segments.copy()
            
            # Merge avec risque churn si disponible
            if churn_predictions is not None:
                df = df.merge(
                    churn_predictions[['customer_id', 'churn_probability', 'risk_category']], 
                    on='customer_id', how='left'
                )
                df['churn_probability'] = df['churn_probability'].fillna(0.1)
                df['risk_category'] = df['risk_category'].fillna('Low')
            
            # =====================================================================
            # 🎯 SEGMENTATION CAMPAGNE AVANCÉE
            # =====================================================================
            
            def assign_campaign_segment(row):
                """Logic de segmentation campagne basée sur RFM + Churn + CLV"""
                
                # High Value Segments
                if (row['monetary'] >= 1000 and row['frequency'] >= 8):
                    if hasattr(row, 'churn_probability') and row.get('churn_probability', 0) > 0.7:
                        return 'VIP_AtRisk'  # Segment critique - intervention urgente
                    else:
                        return 'VIP_Whales'  # Segment premium - fidélisation
                
                # Champions loyaux
                elif (row['rfm_segment'] == 'Champions' and row['recency'] <= 30):
                    return 'Champions_Active'
                
                # Potentiel de croissance
                elif (row['rfm_segment'] in ['Potential Loyalists', 'New Customers'] and 
                      row['monetary'] >= 200):
                    return 'Growth_Potential'
                
                # Récupération urgente
                elif hasattr(row, 'risk_category') and row.get('risk_category') in ['High', 'Critical']:
                    return 'Winback_Urgent'
                
                # Réactivation douce
                elif (row['rfm_segment'] == 'At Risk' and row['recency'] <= 90):
                    return 'Reactivation_Soft'
                
                # Nouveaux clients à développer
                elif row['rfm_segment'] == 'New Customers':
                    return 'New_Development'
                
                # Segment de maintenance/low cost
                else:
                    return 'Maintenance_LowCost'
            
            df['campaign_segment'] = df.apply(assign_campaign_segment, axis=1)
            
            # =====================================================================
            # 📊 ENRICHISSEMENT FEATURES CAMPAGNE
            # =====================================================================
            
            # Scoring de priorité (0-100)
            segment_priorities = {
                'VIP_AtRisk': 100,
                'VIP_Whales': 95,
                'Champions_Active': 90,
                'Growth_Potential': 80,
                'Winback_Urgent': 85,
                'New_Development': 70,
                'Reactivation_Soft': 60,
                'Maintenance_LowCost': 30
            }
            df['campaign_priority'] = df['campaign_segment'].map(segment_priorities)
            
            # Canaux recommandés par segment
            def get_recommended_channels(segment):
                channel_mapping = {
                    'VIP_AtRisk': ['personal_call', 'email', 'direct_mail'],
                    'VIP_Whales': ['email', 'personal_call', 'direct_mail'],
                    'Champions_Active': ['email', 'sms', 'social_media'],
                    'Growth_Potential': ['email', 'social_media', 'google_ads'],
                    'Winback_Urgent': ['sms', 'email', 'display_ads'],
                    'New_Development': ['email', 'social_media', 'google_ads'],
                    'Reactivation_Soft': ['email', 'display_ads'],
                    'Maintenance_LowCost': ['email']
                }
                return channel_mapping.get(segment, ['email'])
            
            df['recommended_channels'] = df['campaign_segment'].apply(get_recommended_channels)
            
            # Budget recommandé par client selon segment
            segment_budgets = {
                'VIP_AtRisk': 50.0,
                'VIP_Whales': 30.0,
                'Champions_Active': 15.0,
                'Growth_Potential': 12.0,
                'Winback_Urgent': 20.0,
                'New_Development': 8.0,
                'Reactivation_Soft': 5.0,
                'Maintenance_LowCost': 2.0
            }
            df['recommended_budget_per_customer'] = df['campaign_segment'].map(segment_budgets)
            
            # Taux de réponse attendu par segment (basé sur l'historique)
            segment_response_rates = {
                'VIP_AtRisk': 0.25,
                'VIP_Whales': 0.35,
                'Champions_Active': 0.40,
                'Growth_Potential': 0.20,
                'Winback_Urgent': 0.15,
                'New_Development': 0.18,
                'Reactivation_Soft': 0.12,
                'Maintenance_LowCost': 0.08
            }
            df['expected_response_rate'] = df['campaign_segment'].map(segment_response_rates)
            
            # ROI estimé par segment
            df['revenue_potential'] = df['monetary'] * 1.2  # 20% uplift estimé
            df['campaign_cost'] = df['recommended_budget_per_customer']
            df['estimated_roi'] = ((df['revenue_potential'] * self.profit_margin * df['expected_response_rate'] - 
                                   df['campaign_cost']) / df['campaign_cost'] * 100).round(1)
            
            # Flag segments profitables
            df['is_profitable_segment'] = (df['estimated_roi'] > 15).astype(int)  # 15% ROI minimum
            
            self.logger.info("✅ Segments campagne créés avec succès")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erreur création segments: {e}")
            raise
    
    def optimize_campaign_budget(self, campaign_segments: pd.DataFrame, 
                               total_budget: float) -> Dict:
        """
        💰 Optimise l'allocation de budget par segment selon ROI
        
        Args:
            campaign_segments: Segments avec métriques campagne
            total_budget: Budget total disponible
            
        Returns:
            Dict avec allocation optimisée
        """
        try:
            self.logger.info(f"💰 Optimisation budget: ${total_budget:,.0f}")
            
            # =====================================================================
            # 📊 ANALYSE PAR SEGMENT
            # =====================================================================
            
            segment_analysis = campaign_segments.groupby('campaign_segment').agg({
                'customer_id': 'count',
                'recommended_budget_per_customer': 'mean',
                'expected_response_rate': 'mean',
                'estimated_roi': 'mean',
                'revenue_potential': 'sum',
                'campaign_priority': 'mean'
            }).round(2)
            
            segment_analysis.columns = ['customers_count', 'avg_cost_per_customer', 
                                      'response_rate', 'roi_percent', 'total_revenue_potential', 'priority']
            
            # Budget total nécessaire par segment
            segment_analysis['total_budget_needed'] = (
                segment_analysis['customers_count'] * segment_analysis['avg_cost_per_customer']
            )
            
            # =====================================================================
            # 🎯 ALLOCATION INTELLIGENTE DU BUDGET
            # =====================================================================
            
            # Tri par priorité et ROI
            segment_analysis['combined_score'] = (
                segment_analysis['priority'] * 0.4 + 
                segment_analysis['roi_percent'] * 0.6
            )
            segment_analysis = segment_analysis.sort_values('combined_score', ascending=False)
            
            # Allocation progressive du budget
            remaining_budget = total_budget
            allocations = {}
            
            for segment, data in segment_analysis.iterrows():
                if remaining_budget <= 0:
                    # Plus de budget disponible
                    allocations[segment] = {
                        'allocated_budget': 0,
                        'customers_targeted': 0,
                        'roi_percent': data['roi_percent'],
                        'priority': data['priority'],
                        'status': 'insufficient_budget'
                    }
                    continue
                
                # Budget nécessaire vs disponible
                needed_budget = data['total_budget_needed']
                
                if needed_budget <= remaining_budget:
                    # Budget suffisant pour tout le segment
                    allocated_budget = needed_budget
                    customers_targeted = data['customers_count']
                    status = 'fully_funded'
                else:
                    # Budget partiel
                    allocated_budget = remaining_budget
                    customers_targeted = int(remaining_budget / data['avg_cost_per_customer'])
                    status = 'partially_funded'
                
                allocations[segment] = {
                    'allocated_budget': round(allocated_budget, 2),
                    'customers_targeted': customers_targeted,
                    'roi_percent': data['roi_percent'],
                    'priority': data['priority'],
                    'expected_revenue': round(customers_targeted * data['total_revenue_potential'] / data['customers_count'] * data['response_rate'], 2),
                    'status': status
                }
                
                remaining_budget -= allocated_budget
            
            # =====================================================================
            # 📈 MÉTRIQUES GLOBALES
            # =====================================================================
            
            total_allocated = sum([alloc['allocated_budget'] for alloc in allocations.values()])
            total_customers_targeted = sum([alloc['customers_targeted'] for alloc in allocations.values()])
            total_expected_revenue = sum([alloc['expected_revenue'] for alloc in allocations.values()])
            
            overall_roi = ((total_expected_revenue * self.profit_margin - total_allocated) / 
                          total_allocated * 100) if total_allocated > 0 else 0
            
            optimization_result = {
                'segment_allocations': allocations,
                'segment_analysis': segment_analysis,
                'total_budget': total_budget,
                'total_allocated': round(total_allocated, 2),
                'remaining_budget': round(remaining_budget, 2),
                'total_customers_targeted': total_customers_targeted,
                'total_expected_revenue': round(total_expected_revenue, 2),
                'overall_roi_percent': round(overall_roi, 1),
                'budget_utilization_percent': round(total_allocated / total_budget * 100, 1),
                'optimization_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            self.logger.info(f"✅ Budget optimisé - ROI: {overall_roi:.1f}%")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"❌ Erreur optimisation budget: {e}")
            raise
    
    def create_campaign_calendar(self, budget_optimization: Dict, 
                               campaign_segments: pd.DataFrame,
                               weeks_ahead: int = 12) -> Tuple[pd.DataFrame, Dict]:
        """
        📅 Crée calendrier optimisé des campagnes
        
        Args:
            budget_optimization: Résultats allocation budget
            campaign_segments: Segments clients
            weeks_ahead: Nombre de semaines à planifier
            
        Returns:
            Tuple[DataFrame calendrier, Dict résumé]
        """
        try:
            self.logger.info(f"📅 Création calendrier {weeks_ahead} semaines...")
            
            # =====================================================================
            # 📅 GÉNÉRATION DU CALENDRIER
            # =====================================================================
            
            start_date = datetime.now()
            calendar_entries = []
            
            # Priorité et timing par type de segment
            segment_timing = {
                'VIP_AtRisk': {'start_week': 1, 'frequency': 2, 'duration': 8},      # Urgent
                'Winback_Urgent': {'start_week': 1, 'frequency': 3, 'duration': 6}, # Urgent
                'VIP_Whales': {'start_week': 2, 'frequency': 4, 'duration': 12},    # Fidélisation continue
                'Champions_Active': {'start_week': 3, 'frequency': 4, 'duration': 10},
                'Growth_Potential': {'start_week': 4, 'frequency': 3, 'duration': 8},
                'New_Development': {'start_week': 5, 'frequency': 2, 'duration': 6},
                'Reactivation_Soft': {'start_week': 6, 'frequency': 5, 'duration': 8},
                'Maintenance_LowCost': {'start_week': 8, 'frequency': 8, 'duration': 4}
            }
            
            campaign_id = 1
            
            for segment, allocation in budget_optimization['segment_allocations'].items():
                if allocation['customers_targeted'] == 0:
                    continue
                
                timing = segment_timing.get(segment, {'start_week': 4, 'frequency': 4, 'duration': 6})
                
                # Génération des campagnes pour ce segment
                current_week = timing['start_week']
                campaigns_created = 0
                max_campaigns = min(timing['duration'] // timing['frequency'], 
                                  weeks_ahead // timing['frequency'])
                
                while current_week <= weeks_ahead and campaigns_created < max_campaigns:
                    
                    campaign_date = start_date + timedelta(weeks=current_week-1)
                    
                    # Répartition des clients si budget limité
                    customers_per_campaign = min(
                        allocation['customers_targeted'] // max_campaigns,
                        allocation['customers_targeted']
                    )
                    
                    if customers_per_campaign == 0:
                        break
                    
                    budget_per_campaign = customers_per_campaign * (allocation['allocated_budget'] / allocation['customers_targeted'])
                    
                    # Canaux pour ce segment
                    segment_data = campaign_segments[campaign_segments['campaign_segment'] == segment]
                    if len(segment_data) > 0:
                        channels = segment_data.iloc[0]['recommended_channels'][:2]  # Top 2 channels
                    else:
                        channels = ['email']
                    
                    calendar_entries.append({
                        'campaign_id': f"CAMP_{campaign_id:04d}",
                        'campaign_name': f"{segment}_Week{current_week}",
                        'segment': segment,
                        'start_date': campaign_date.strftime('%Y-%m-%d'),
                        'week_number': current_week,
                        'channels': ', '.join(channels),
                        'customers_targeted': customers_per_campaign,
                        'budget_allocated': round(budget_per_campaign, 2),
                        'expected_roi_percent': allocation['roi_percent'],
                        'priority_level': allocation['priority'],
                        'campaign_type': self._get_campaign_type(segment),
                        'status': 'planned'
                    })
                    
                    campaign_id += 1
                    current_week += timing['frequency']
                    campaigns_created += 1
            
            # =====================================================================
            # 📊 DATAFRAME CALENDRIER
            # =====================================================================
            
            calendar_df = pd.DataFrame(calendar_entries)
            
            if len(calendar_df) > 0:
                # Tri par semaine et priorité
                calendar_df = calendar_df.sort_values(['week_number', 'priority_level'], 
                                                    ascending=[True, False])
                
                # Calculs de performance
                calendar_df['expected_revenue'] = (
                    calendar_df['customers_targeted'] * 
                    calendar_df['budget_allocated'] * 
                    calendar_df['expected_roi_percent'] / 100
                ).round(2)
                
                # Ajout colonnes utiles
                calendar_df['quarter'] = calendar_df['week_number'].apply(
                    lambda x: f"Q{(x-1)//13 + 1}" if x <= 52 else "Q1"
                )
                calendar_df['month'] = pd.to_datetime(calendar_df['start_date']).dt.strftime('%Y-%m')
            
            # =====================================================================
            # 📈 RÉSUMÉ CALENDRIER
            # =====================================================================
            
            if len(calendar_df) > 0:
                
                monthly_summary = calendar_df.groupby('month').agg({
                    'campaign_id': 'count',
                    'customers_targeted': 'sum',
                    'budget_allocated': 'sum',
                    'expected_revenue': 'sum'
                }).round(2)
                
                calendar_summary = {
                    'total_campaigns': len(calendar_df),
                    'total_weeks_covered': calendar_df['week_number'].max(),
                    'total_budget_scheduled': calendar_df['budget_allocated'].sum(),
                    'total_customers_reached': calendar_df['customers_targeted'].sum(),
                    'total_expected_revenue': calendar_df['expected_revenue'].sum(),
                    'campaigns_per_month': monthly_summary.to_dict(),
                    'top_segment_campaigns': calendar_df['segment'].value_counts().head(3).to_dict(),
                    'avg_budget_per_campaign': round(calendar_df['budget_allocated'].mean(), 2),
                    'calendar_created': datetime.now().strftime('%Y-%m-%d')
                }
                
            else:
                calendar_summary = {'total_campaigns': 0, 'message': 'Aucune campagne planifiée'}
            
            self.logger.info(f"✅ Calendrier créé - {len(calendar_df)} campagnes")
            
            return calendar_df, calendar_summary
            
        except Exception as e:
            self.logger.error(f"❌ Erreur création calendrier: {e}")
            raise
    
    def _get_campaign_type(self, segment: str) -> str:
        """Détermine le type de campagne selon le segment"""
        
        type_mapping = {
            'VIP_AtRisk': 'retention_urgent',
            'VIP_Whales': 'loyalty_premium', 
            'Champions_Active': 'engagement_loyalty',
            'Growth_Potential': 'upsell_cross_sell',
            'Winback_Urgent': 'winback_aggressive',
            'New_Development': 'onboarding_nurture',
            'Reactivation_Soft': 'reactivation_gentle',
            'Maintenance_LowCost': 'maintenance_basic'
        }
        
        return type_mapping.get(segment, 'standard_campaign')
    
    def simulate_ab_test(self, test_segment: pd.DataFrame, 
                        test_config: Dict) -> Dict:
        """
        🧪 Simule un test A/B sur un segment
        
        Args:
            test_segment: Segment de clients à tester
            test_config: Configuration du test {variant_A: {...}, variant_B: {...}}
            
        Returns:
            Dict avec résultats du test A/B
        """
        try:
            self.logger.info("🧪 Simulation test A/B...")
            
            if len(test_segment) < 10:
                return {'error': 'Segment trop petit pour test A/B'}
            
            # =====================================================================
            # 🎯 SETUP TEST
            # =====================================================================
            
            # Division 50/50 aléatoire
            np.random.seed(42)  # Reproductibilité
            test_segment_copy = test_segment.copy()
            test_segment_copy['variant'] = np.random.choice(['A', 'B'], len(test_segment_copy))
            
            variant_a = test_segment_copy[test_segment_copy['variant'] == 'A']
            variant_b = test_segment_copy[test_segment_copy['variant'] == 'B']
            
            # Configuration par défaut
            config_a = test_config.get('variant_A', {})
            config_b = test_config.get('variant_B', {})
            
            # =====================================================================
            # 📊 SIMULATION RÉSULTATS
            # =====================================================================
            
            def simulate_variant_performance(segment_df, config):
                """Simule performance d'une variante"""
                
                base_response_rate = segment_df['expected_response_rate'].mean()
                base_revenue_per_customer = segment_df['revenue_potential'].mean()
                base_cost_per_customer = segment_df['recommended_budget_per_customer'].mean()
                
                # Application des lifts
                response_lift = config.get('response_lift', 0)
                conversion_lift = config.get('conversion_lift', 0) 
                cost_modifier = config.get('cost_modifier', 1.0)
                
                actual_response_rate = base_response_rate * (1 + response_lift)
                actual_revenue_per_customer = base_revenue_per_customer * (1 + conversion_lift)
                actual_cost_per_customer = base_cost_per_customer * cost_modifier
                
                # Simulation avec variabilité réaliste
                n_customers = len(segment_df)
                
                # Réponses (binomial)
                responses = np.random.binomial(1, actual_response_rate, n_customers)
                n_responses = responses.sum()
                
                # Revenus (pour ceux qui répondent)
                revenues = np.random.normal(actual_revenue_per_customer, 
                                          actual_revenue_per_customer * 0.3, 
                                          n_responses)
                revenues = np.maximum(revenues, 0)  # Pas de revenus négatifs
                
                total_revenue = revenues.sum()
                total_cost = n_customers * actual_cost_per_customer
                roi = ((total_revenue * self.profit_margin - total_cost) / total_cost * 100) if total_cost > 0 else 0
                
                return {
                    'customers': n_customers,
                    'responders': n_responses,
                    'response_rate': n_responses / n_customers,
                    'total_revenue': total_revenue,
                    'avg_revenue_per_responder': revenues.mean() if n_responses > 0 else 0,
                    'total_cost': total_cost,
                    'roi_percent': roi,
                    'profit': total_revenue * self.profit_margin - total_cost
                }
            
            # Simulation des deux variantes
            results_a = simulate_variant_performance(variant_a, config_a)
            results_b = simulate_variant_performance(variant_b, config_b)
            
            # =====================================================================
            # 📈 TESTS STATISTIQUES
            # =====================================================================
            
            # Test de différence de taux de réponse
            response_rate_diff = results_b['response_rate'] - results_a['response_rate']
            
            # Test z pour différence de proportions
            p1, p2 = results_a['response_rate'], results_b['response_rate']
            n1, n2 = results_a['customers'], results_b['customers']
            
            if p1 > 0 and p2 > 0 and n1 > 0 and n2 > 0:
                p_pool = (results_a['responders'] + results_b['responders']) / (n1 + n2)
                se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
                z_score = response_rate_diff / se if se > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                is_significant = p_value < 0.05
                confidence_level = 95
                
            else:
                z_score = 0
                p_value = 1.0
                is_significant = False
                confidence_level = 0
            
            # =====================================================================
            # 🏆 RÉSULTATS FINAUX
            # =====================================================================
            
            # Détermination du gagnant
            if results_b['roi_percent'] > results_a['roi_percent']:
                winner = 'B'
                winner_lift = ((results_b['roi_percent'] - results_a['roi_percent']) / 
                              results_a['roi_percent'] * 100) if results_a['roi_percent'] > 0 else 0
            elif results_a['roi_percent'] > results_b['roi_percent']:
                winner = 'A'
                winner_lift = ((results_a['roi_percent'] - results_b['roi_percent']) / 
                              results_b['roi_percent'] * 100) if results_b['roi_percent'] > 0 else 0
            else:
                winner = 'tie'
                winner_lift = 0
            
            ab_test_results = {
                'test_setup': {
                    'total_customers': len(test_segment),
                    'variant_a_customers': len(variant_a),
                    'variant_b_customers': len(variant_b),
                    'test_configuration': test_config,
                    'test_date': datetime.now().strftime('%Y-%m-%d')
                },
                
                'variant_a_results': results_a,
                'variant_b_results': results_b,
                
                'statistical_analysis': {
                    'response_rate_diff': round(response_rate_diff, 4),
                    'z_score': round(z_score, 3),
                    'p_value': round(p_value, 4),
                    'is_significant': is_significant,
                    'confidence_level': confidence_level
                },
                
                'conclusion': {
                    'winner': winner,
                    'winner_lift_percent': round(winner_lift, 1),
                    'roi_difference': round(results_b['roi_percent'] - results_a['roi_percent'], 1),
                    'revenue_difference': round(results_b['total_revenue'] - results_a['total_revenue'], 2),
                    'recommend_variant': winner if is_significant else 'A',  # Default à A si non significatif
                    'statistical_significance': 'Significant' if is_significant else 'Not Significant'
                }
            }
            
            self.logger.info(f"✅ Test A/B simulé - Gagnant: {winner}")
            
            return ab_test_results
            
        except Exception as e:
            self.logger.error(f"❌ Erreur simulation A/B: {e}")
            raise
    
    def calculate_campaign_attribution(self, campaign_data: pd.DataFrame,
                                     attribution_model: str = 'last_touch') -> Tuple[pd.DataFrame, Dict]:
        """
        📊 Calcule l'attribution multi-canal des campagnes
        
        Args:
            campaign_data: Données campagnes avec customer_id et revenus
            attribution_model: 'last_touch', 'first_touch', 'linear', 'time_decay'
            
        Returns:
            Tuple[Attribution par canal, Résumé attribution]
        """
        try:
            self.logger.info(f"📊 Attribution {attribution_model}...")
            
            # =====================================================================
            # 🎯 SIMULATION TOUCHPOINTS MARKETING
            # =====================================================================
            
            attribution_results = []
            
            # Simulation des parcours clients multi-touchpoints
            for _, row in campaign_data.iterrows():
                
                # Simulation touchpoints réalistes selon segment
                segment = row.get('campaign_segment', 'standard')
                revenue = row.get('revenue', row.get('monetary', 0))
                
                if revenue <= 0:
                    continue
                
                # Génération touchpoints selon le segment
                if segment in ['VIP_Whales', 'VIP_AtRisk']:
                    # Parcours complexe pour VIPs
                    possible_touchpoints = ['email', 'personal_call', 'direct_mail', 'social_media']
                    n_touchpoints = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
                
                elif segment in ['Growth_Potential', 'Champions_Active']:
                    # Parcours moyen
                    possible_touchpoints = ['email', 'social_media', 'google_ads', 'sms']
                    n_touchpoints = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                
                else:
                    # Parcours simple
                    possible_touchpoints = ['email', 'social_media', 'google_ads']
                    n_touchpoints = np.random.choice([1, 2], p=[0.7, 0.3])
                
                # Sélection touchpoints
                marketing_touchpoints = np.random.choice(
                    possible_touchpoints, 
                    size=min(n_touchpoints, len(possible_touchpoints)), 
                    replace=False
                ).tolist()
                
                # =====================================================================
                # 🔄 MODÈLES D'ATTRIBUTION
                # =====================================================================
                
                
                if attribution_model == 'last_touch':

                    # 100% au dernier touchpoint
