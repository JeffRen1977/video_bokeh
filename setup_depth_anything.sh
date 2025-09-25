#!/bin/bash

# Setup script for Depth-Anything repository
echo "🔧 Setting up Depth-Anything repository..."

# Clone Depth-Anything repository if it doesn't exist
if [ ! -d "Depth-Anything" ]; then
    echo "📥 Cloning Depth-Anything repository..."
    git clone https://github.com/LiheYoung/Depth-Anything.git
    
    if [ $? -eq 0 ]; then
        echo "✅ Depth-Anything repository cloned successfully!"
    else
        echo "❌ Failed to clone Depth-Anything repository"
        exit 1
    fi
else
    echo "✅ Depth-Anything repository already exists"
fi

# Install Depth-Anything requirements
echo "📦 Installing Depth-Anything requirements..."
cd Depth-Anything
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Depth-Anything requirements installed successfully!"
else
    echo "⚠️  Some Depth-Anything requirements may have failed to install"
    echo "   This is normal if you already have the required packages"
fi

cd ..

echo ""
echo "🎉 Setup complete! You can now run:"
echo "   python portrait_comparison.py --compare"
echo ""
