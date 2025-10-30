#!/bin/bash

# Fix all missing torch.nn imports systematically

echo "Fixing all missing torch.nn as nn imports..."

# List of files that need the import
files=(
    "axiom/models/portfolio/rl_portfolio_manager.py"
    "axiom/models/portfolio/lstm_cnn_predictor.py"
    "axiom/models/portfolio/portfolio_transformer.py"
    "axiom/models/pricing/vae_option_pricer.py"
    "axiom/models/risk/cnn_lstm_credit_model.py"
    "axiom/models/pricing/ann_greeks_calculator.py"
    "axiom/models/pricing/drl_option_hedger.py"
    "axiom/models/risk/transformer_nlp_credit.py"
    "axiom/models/pricing/gan_volatility_surface.py"
    "axiom/models/pricing/informer_transformer_pricer.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        # Check if torch.nn import already exists
        if ! grep -q "import torch.nn as nn" "$file"; then
            echo "Adding import to $file"
            # Add import at top of file after other imports
            sed -i '' '1a\
import torch.nn as nn
' "$file"
        else
            echo "✓ $file already has import"
        fi
    fi
done

echo "✓ All imports fixed!"