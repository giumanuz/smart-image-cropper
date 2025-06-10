#!/bin/bash

# Smart Image Cropper - Development Setup Script

echo "🚀 Setting up Smart Image Cropper development environment..."

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | cut -d" " -f2 | cut -d"." -f1-2)
echo "📍 Detected Python version: $python_version"

if [[ $(echo "$python_version < 3.8" | bc) -eq 0 ]]; then
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install build tools
echo "🛠️  Installing build tools..."
pip install --upgrade setuptools wheel twine build

# Install package in development mode
echo "📋 Installing package in development mode..."
pip install -e ".[dev]"

# Run tests to verify setup
echo "🧪 Running tests to verify setup..."
if command -v pytest &>/dev/null; then
    pytest tests/ -v
    if [ $? -eq 0 ]; then
        echo "✅ All tests passed!"
    else
        echo "⚠️  Some tests failed, but setup is complete."
    fi
else
    echo "⚠️  pytest not found, skipping tests."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Update your API credentials in the example files"
echo "3. Run the example: python examples/basic_usage.py"
echo "4. Build the package: python -m build"
echo "5. Check the publishing guide: PUBLISHING_GUIDE.md"
echo ""
echo "Happy coding! 🐍✨"
