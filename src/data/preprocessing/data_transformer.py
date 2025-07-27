# src/data/data_transformer.py
"""
🔧 Data Transformer - Pipeline de nettoyage e-commerce
🎯 Basé sur votre notebook d'analyse exploratoire
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from config.settings import settings, get_logger

logger = get_logger(__name__)

@dataclass
class CleanDataResult:
    """📊 Résultat du nettoyage des données"""
    data: pd.DataFrame
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    cleaning_time_seconds: float
    transformations_applied: List[str]
    data_quality_score: float
    memory_usage_mb: float

class EcommerceDataTransformer:
    """
    🔧 Transformateur de données e-commerce
    
    Pipeline basé sur votre notebook:
    1. Gestion valeurs manquantes
    2. Suppression doublons
    3. Parsing dates
    4. Détection retours
    5. Validation prix/quantités
    6. Optimisation types
    """
    
    def __init__(self, 
                 remove_duplicates: bool = True,
                 handle_missing: bool = True,
                 parse_dates: bool = True,
                 detect_returns: bool = True):
        
        self.remove_duplicates = remove_duplicates
        self.handle_missing = handle_missing
        self.parse_dates = parse_dates
        self.detect_returns = detect_returns
        
        logger.info("🔧 DataTransformer initialisé")
    
    def clean_dataset(self, 
                     raw_data: pd.DataFrame,
                     preserve_original: bool = True) -> CleanDataResult:
        """
        🧹 Pipeline complet de nettoyage
        
        Args:
            raw_data: DataFrame brut depuis data_loader
            preserve_original: Garder une copie du DataFrame original
            
        Returns:
            CleanDataResult avec données nettoyées et métriques
        """
        start_time = datetime.now()
        
        if preserve_original:
            df = raw_data.copy()
        else:
            df = raw_data
        
        # Métriques initiales
        rows_before = len(df)
        columns_before = len(df.columns)
        transformations = []
        
        logger.info(f"🧹 Début nettoyage: {rows_before} lignes, {columns_before} colonnes")
        
        # Pipeline de nettoyage
        if self.handle_missing and not df.empty:
            df, missing_transf = self._handle_missing_values(df)
            transformations.extend(missing_transf)
        
        if self.remove_duplicates and not df.empty:
            df, dup_transf = self._remove_duplicates(df)
            transformations.extend(dup_transf)
        
        if self.parse_dates and not df.empty:
            df, date_transf = self._parse_invoice_dates(df)
            transformations.extend(date_transf)
        
        if self.detect_returns and not df.empty:
            df, return_transf = self._detect_returns(df)
            transformations.extend(return_transf)
        
        if not df.empty:
            df, validation_transf = self._validate_business_rules(df)
            transformations.extend(validation_transf)
            
            df, optimize_transf = self._optimize_data_types(df)
            transformations.extend(optimize_transf)
        
        # Métriques finales
        rows_after = len(df)
        columns_after = len(df.columns)
        cleaning_time = (datetime.now() - start_time).total_seconds()
        
        # Calcul score qualité
        quality_score = self._calculate_quality_score(df)
        
        # Usage mémoire
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        result = CleanDataResult(
            data=df,
            rows_before=rows_before,
            rows_after=rows_after,
            columns_before=columns_before,
            columns_after=columns_after,
            cleaning_time_seconds=cleaning_time,
            transformations_applied=transformations,
            data_quality_score=quality_score,
            memory_usage_mb=memory_mb
        )
        
        logger.info(f"✅ Nettoyage terminé: {rows_after} lignes (-{rows_before-rows_after}), "
                   f"score qualité: {quality_score:.2f}, {cleaning_time:.2f}s")
        
        return result
    
    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """🕳️ Gestion des valeurs manquantes selon logique e-commerce"""
        transformations = []
        initial_nulls = df.isnull().sum().sum()
        
        if initial_nulls == 0:
            return df, transformations
        
        logger.info(f"🕳️ Traitement {initial_nulls} valeurs manquantes")
        
        # Customer ID: Garder les NaN (transactions sans compte client)
        if 'Customer ID' in df.columns:
            customer_nulls = df['Customer ID'].isnull().sum()
            logger.info(f"📊 {customer_nulls} transactions sans Customer ID (gardées)")
            transformations.append("customer_id_nulls_preserved")
        
        # Description: Supprimer si manquante (produit non identifiable)
        if 'Description' in df.columns:
            desc_nulls_before = df['Description'].isnull().sum()
            if desc_nulls_before > 0:
                df = df.dropna(subset=['Description'])
                logger.info(f"🗑️ Supprimé {desc_nulls_before} lignes sans Description")
                transformations.append("missing_descriptions_removed")
        
        # Invoice: Supprimer si manquant (transaction invalide)
        if 'Invoice' in df.columns:
            invoice_nulls = df['Invoice'].isnull().sum()
            if invoice_nulls > 0:
                df = df.dropna(subset=['Invoice'])
                logger.info(f"🗑️ Supprimé {invoice_nulls} lignes sans Invoice")
                transformations.append("missing_invoices_removed")
        
        # StockCode: Supprimer si manquant (produit non identifiable)
        if 'StockCode' in df.columns:
            stock_nulls = df['StockCode'].isnull().sum()
            if stock_nulls > 0:
                df = df.dropna(subset=['StockCode'])
                logger.info(f"🗑️ Supprimé {stock_nulls} lignes sans StockCode")
                transformations.append("missing_stockcodes_removed")
        
        # Price: Supprimer si manquant ou zéro (transaction invalide)
        if 'Price' in df.columns:
            price_issues = df['Price'].isnull() | (df['Price'] == 0)
            price_count = price_issues.sum()
            if price_count > 0:
                df = df[~price_issues]
                logger.info(f"🗑️ Supprimé {price_count} lignes avec Price manquant/zéro")
                transformations.append("invalid_prices_removed")
        
        # Quantity: Supprimer si manquant (mais garder les négatifs = retours)
        if 'Quantity' in df.columns:
            qty_nulls = df['Quantity'].isnull().sum()
            if qty_nulls > 0:
                df = df.dropna(subset=['Quantity'])
                logger.info(f"🗑️ Supprimé {qty_nulls} lignes sans Quantity")
                transformations.append("missing_quantities_removed")
        
        final_nulls = df.isnull().sum().sum()
        logger.info(f"✅ Valeurs manquantes: {initial_nulls} → {final_nulls}")
        
        if len(transformations) > 0:
            transformations.append("missing_values_handled")
        
        return df, transformations
    
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """🔄 Suppression des doublons complets"""
        transformations = []
        duplicates_before = df.duplicated().sum()
        
        if duplicates_before == 0:
            return df, transformations
        
        logger.info(f"🔄 Suppression de {duplicates_before} doublons")
        
        df_clean = df.drop_duplicates()
        duplicates_removed = duplicates_before
        
        logger.info(f"✅ Doublons supprimés: {duplicates_removed}")
        transformations.append("duplicates_removed")
        
        return df_clean, transformations
    
    def _parse_invoice_dates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """📅 Parsing des dates InvoiceDate"""
        transformations = []
        
        if 'InvoiceDate' not in df.columns:
            return df, transformations
        
        logger.info("📅 Parsing des dates InvoiceDate")
        
        try:
            # Format typique: "12/1/2010 8:26"
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M', errors='coerce')
            
            # Vérifier les dates non parsées
            invalid_dates = df['InvoiceDate'].isnull().sum()
            if invalid_dates > 0:
                logger.warning(f"⚠️ {invalid_dates} dates non parsables (gardées comme NaT)")
            
            # Extraction de features temporelles
            df['Year'] = df['InvoiceDate'].dt.year
            df['Month'] = df['InvoiceDate'].dt.month
            df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
            df['Hour'] = df['InvoiceDate'].dt.hour
            
            logger.info("✅ Dates parsées + features temporelles créées")
            transformations.extend(["date_parsing", "temporal_features_created"])
            
        except Exception as e:
            logger.error(f"❌ Erreur parsing dates: {e}")
            transformations.append("date_parsing_failed")
        
        return df, transformations
    
    def _detect_returns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """🔄 Détection des retours (Invoice commençant par C)"""
        transformations = []
        
        if 'Invoice' not in df.columns:
            return df, transformations
        
        logger.info("🔄 Détection des retours")
        
        # Retours = Invoice commençant par 'C'
        df['IsReturn'] = df['Invoice'].astype(str).str.startswith('C')
        returns_count = df['IsReturn'].sum()
        
        # Nettoyage Invoice pour retours (enlever le C)
        df['InvoiceClean'] = df['Invoice'].astype(str).str.replace('^C', '', regex=True)
        
        logger.info(f"✅ {returns_count} retours détectés ({returns_count/len(df)*100:.1f}%)")
        transformations.extend(["returns_detected", "invoice_cleaned"])
        
        return df, transformations
    
    def _validate_business_rules(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """🏪 Validation des règles métier e-commerce"""
        transformations = []
        initial_rows = len(df)
        
        logger.info("🏪 Validation des règles métier")
        
        # Règle 1: Quantité cohérente avec type de transaction
        if 'Quantity' in df.columns and 'IsReturn' in df.columns:
            # Retours doivent avoir quantité négative (normalement)
            inconsistent = (df['IsReturn'] == True) & (df['Quantity'] > 0)
            inconsistent_count = inconsistent.sum()
            
            if inconsistent_count > 0:
                logger.info(f"📊 {inconsistent_count} retours avec quantité positive (gardés)")
                transformations.append("return_quantity_inconsistency_noted")
        
        # Règle 2: Prix unitaire raisonnable (> 0, < seuil)
        if 'Price' in df.columns:
            max_reasonable_price = 1000  # £1000 max par unité
            expensive_items = df['Price'] > max_reasonable_price
            expensive_count = expensive_items.sum()
            
            if expensive_count > 0:
                logger.info(f"💰 {expensive_count} items > £{max_reasonable_price} (gardés)")
                transformations.append("expensive_items_flagged")
        
        # Règle 3: Cohérence Customer ID (doit être numérique si présent)
        if 'Customer ID' in df.columns:
            # Garder seulement les ID numériques valides ou NaN
            valid_customers = df['Customer ID'].isnull() | (df['Customer ID'] > 0)
            invalid_customers = (~valid_customers).sum()
            
            if invalid_customers > 0:
                df = df[valid_customers]
                logger.info(f"🗑️ Supprimé {invalid_customers} Customer ID invalides")
                transformations.append("invalid_customer_ids_removed")
        
        # Règle 4: Calcul du montant total
        if 'Price' in df.columns and 'Quantity' in df.columns:
            df['TotalAmount'] = df['Price'] * df['Quantity']
            transformations.append("total_amount_calculated")
        
        rows_after_validation = len(df)
        removed_rows = initial_rows - rows_after_validation
        
        if removed_rows > 0:
            logger.info(f"🗑️ {removed_rows} lignes supprimées par validation métier")
        
        transformations.append("business_rules_validated")
        return df, transformations
    
    def _optimize_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """⚡ Optimisation des types de données pour la mémoire"""
        transformations = []
        memory_before = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        logger.info(f"⚡ Optimisation types (mémoire avant: {memory_before:.1f}MB)")
        
        # Customer ID: float32 (car peut avoir NaN)
        if 'Customer ID' in df.columns:
            df['Customer ID'] = pd.to_numeric(df['Customer ID'], errors='coerce').astype('float32')
        
        # Price: float32 (suffisant pour prix)
        if 'Price' in df.columns:
            df['Price'] = df['Price'].astype('float32')
        
        # Quantity: int16 (quantités rarements > 32k)
        if 'Quantity' in df.columns:
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').astype('int32')
        
        # TotalAmount: float32
        if 'TotalAmount' in df.columns:
            df['TotalAmount'] = df['TotalAmount'].astype('float32')
        
        # Colonnes temporelles: uint8/uint16
        if 'Year' in df.columns:
            df['Year'] = df['Year'].astype('uint16')
        if 'Month' in df.columns:
            df['Month'] = df['Month'].astype('uint8')
        if 'DayOfWeek' in df.columns:
            df['DayOfWeek'] = df['DayOfWeek'].astype('uint8')
        if 'Hour' in df.columns:
            df['Hour'] = df['Hour'].astype('uint8')
        
        # Colonnes texte: category si peu de valeurs uniques
        text_columns = ['StockCode', 'Country', 'Invoice', 'InvoiceClean']
        for col in text_columns:
            if col in df.columns:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Si moins de 50% de valeurs uniques
                    df[col] = df[col].astype('category')
                    transformations.append(f"{col}_to_category")
        
        # Boolean: bool
        if 'IsReturn' in df.columns:
            df['IsReturn'] = df['IsReturn'].astype('bool')
        
        memory_after = df.memory_usage(deep=True).sum() / 1024 / 1024
        memory_saved = memory_before - memory_after
        
        logger.info(f"✅ Optimisation terminée: {memory_after:.1f}MB (-{memory_saved:.1f}MB, {memory_saved/memory_before*100:.1f}%)")
        transformations.append("data_types_optimized")
        
        return df, transformations
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """📊 Calcul du score de qualité des données (0-100)"""
        if df.empty:
            return 0.0
        
        score_components = []
        
        # 1. Complétude (colonnes critiques sans NaN)
        critical_cols = ['Invoice', 'StockCode', 'Description', 'Price', 'Quantity']
        existing_critical = [col for col in critical_cols if col in df.columns]
        
        if existing_critical:
            completeness = 1 - (df[existing_critical].isnull().sum().sum() / 
                               (len(df) * len(existing_critical)))
            score_components.append(completeness * 30)  # 30% du score
        
        # 2. Unicité (pas de doublons)
        uniqueness = 1 - (df.duplicated().sum() / len(df))
        score_components.append(uniqueness * 20)  # 20% du score
        
        # 3. Validité (prix > 0, quantités cohérentes)
        validity_score = 1.0
        if 'Price' in df.columns:
            invalid_prices = (df['Price'] <= 0).sum()
            validity_score *= (1 - invalid_prices / len(df))
        
        score_components.append(validity_score * 25)  # 25% du score
        
        # 4. Consistance (dates parsables, types corrects)
        consistency_score = 1.0
        if 'InvoiceDate' in df.columns:
            invalid_dates = df['InvoiceDate'].isnull().sum()
            consistency_score *= (1 - invalid_dates / len(df))
        
        score_components.append(consistency_score * 25)  # 25% du score
        
        # Score final
        final_score = sum(score_components)
        return min(100.0, max(0.0, final_score))  # Entre 0 et 100

def create_data_transformer(**kwargs) -> EcommerceDataTransformer:
    """🏭 Factory function avec configuration"""
    return EcommerceDataTransformer(**kwargs)
