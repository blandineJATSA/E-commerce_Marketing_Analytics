# Test avec vos données
from src.data.data_loader import create_data_loader
from src.data.data_transformer import create_data_transformer

# Charger
loader = create_data_loader()
raw_data = loader.load_csv("online_retail_II.csv")

# Transformer  
transformer = create_data_transformer()
clean_result = transformer.clean_dataset(raw_data.data)

print(f"📊 {clean_result.rows_before} → {clean_result.rows_after} lignes")
print(f"⚡ Score qualité: {clean_result.data_quality_score:.1f}/100")
print(f"🔧 Transformations: {clean_result.transformations_applied}")
