# tests/run_tests.sh
#!/bin/bash

echo "🧪 LANCEMENT DES TESTS E-COMMERCE ANALYTICS"
echo "=========================================="

echo ""
echo "📋 1. Tests unitaires rapides..."
pytest tests/unit/ -v --tb=short

echo ""
echo "🔗 2. Tests d'intégration..."
pytest tests/integration/test_full_pipeline.py::TestFullDataPipeline -v --tb=short

echo ""
echo "⚙️ 3. Tests de configuration..."
pytest tests/integration/test_full_pipeline.py::TestPipelineConfigurations -v --tb=short

echo ""
echo "🐘 4. Tests longs (optionnel)..."
read -p "Lancer les tests longs? (y/N): " -n 1 -r
echo
if [[  $ REPLY =~ ^[Yy] $  ]]; then
    pytest tests/integration/ -v -m "slow" --tb=short
else
    echo "⏩ Tests longs skippés"
fi

echo ""
echo "📊 5. Rapport de couverture..."
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

echo ""
echo "✅ Tests terminés! Voir htmlcov/index.html pour le rapport détaillé"
