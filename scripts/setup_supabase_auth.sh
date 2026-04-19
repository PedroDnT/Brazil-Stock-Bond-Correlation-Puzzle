#!/bin/bash
# Setup script for Supabase authentication automation

set -e  # Exit on error

echo "=== Supabase Auth Agent Setup ==="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not detected."
    echo "Activating .venv..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "❌ .venv not found. Creating virtual environment..."
        python3 -m venv .venv
        source .venv/bin/activate
    fi
fi

echo "✓ Using Python: $(which python)"
echo ""

# Install required packages
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install playwright supabase python-dotenv

echo ""
echo "🌐 Installing Chromium for Playwright..."
playwright install chromium

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Make sure you've visited https://www.painelfidc.com.br/dataset-fidc"
echo "  2. Run: python supabase_auth_agent.py"
echo "  3. Or run: python fetch_supabase.py (to authenticate + fetch data)"
echo ""
echo "For more info, see SUPABASE_AUTH_README.md"
