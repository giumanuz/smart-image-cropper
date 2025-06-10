# Publishing Smart Image Cropper to PyPI

This guide will walk you through the process of publishing your Smart Image
Cropper library to PyPI (Python Package Index).

## Prerequisites

1. **Python 3.8+** installed
2. **Git** installed and configured
3. **PyPI account** - Sign up at
   [https://pypi.org/account/register/](https://pypi.org/account/register/)
4. **TestPyPI account** (optional but recommended) - Sign up at
   [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/)

## Step 1: Set Up Your Development Environment

```bash
# Clone your repository (or navigate to your project directory)
cd smart-image-cropper

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install build tools
pip install --upgrade pip setuptools wheel twine build

# Install your package in development mode
pip install -e ".[dev]"
```

## Step 2: Update Package Information

Before publishing, make sure to update the following files with your actual
information:

### 1. Update `setup.py` and `pyproject.toml`

Replace placeholder information:

- `author="Your Name"`
- `author_email="your.email@example.com"`
- `url="https://github.com/yourusername/smart-image-cropper"`
- All GitHub URLs with your actual repository URLs

### 2. Update `smart_image_cropper/__init__.py`

- Update `__author__` and `__email__` fields

## Step 3: Test Your Package Locally

```bash
# Run tests (if you have them)
pytest

# Check if your package can be built
python -m build

# This should create files in dist/ directory:
# - smart_image_cropper-1.0.0-py3-none-any.whl
# - smart_image_cropper-1.0.0.tar.gz
```

## Step 4: Validate Your Package

```bash
# Check your package with twine
twine check dist/*

# This should show "PASSED" for all checks
```

## Step 5: Test on TestPyPI (Recommended)

TestPyPI is a separate instance of PyPI for testing packages.

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# You'll be prompted for your TestPyPI credentials
# Username: your-testpypi-username
# Password: your-testpypi-password (or token)
```

### Test Installation from TestPyPI

```bash
# Create a new environment to test installation
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ smart-image-cropper

# Test that it works
python -c "from smart_image_cropper import SmartImageCropper; print('Import successful!')"
```

## Step 6: Publish to PyPI

Once you've tested on TestPyPI and everything works:

```bash
# Upload to the real PyPI
twine upload dist/*

# Enter your PyPI credentials when prompted
```

## Step 7: Set Up API Tokens (Recommended)

For better security, use API tokens instead of username/password:

1. Go to
   [https://pypi.org/manage/account/token/](https://pypi.org/manage/account/token/)
2. Create a new token with scope limited to your project
3. Configure your `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-testpypi-token-here
```

## Step 8: Automate with GitHub Actions (Optional)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.8"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

Then add your PyPI API token as a secret in your GitHub repository:

1. Go to your repo settings → Secrets and variables → Actions
2. Add a new secret named `PYPI_API_TOKEN` with your token value

## Step 9: Update Version for Future Releases

For subsequent releases:

1. **Update version number** in:

   - `setup.py`
   - `pyproject.toml`
   - `smart_image_cropper/__init__.py`

2. **Clean previous builds**:

   ```bash
   rm -rf dist/ build/ *.egg-info/
   ```

3. **Build and upload new version**:
   ```bash
   python -m build
   twine upload dist/*
   ```

## Common Issues and Solutions

### Issue: Package name already exists

- Choose a different name or add a prefix/suffix
- Update the `name` field in `setup.py` and `pyproject.toml`

### Issue: Upload fails with authentication error

- Make sure you're using the correct credentials
- Try using API tokens instead of username/password
- Check if 2FA is enabled on your account

### Issue: "File already exists" error

- You're trying to upload the same version twice
- Update the version number and rebuild

### Issue: Import errors after installation

- Check that all dependencies are correctly specified
- Verify the package structure matches the imports

## Verification After Publishing

1. **Check your package page**: `https://pypi.org/project/smart-image-cropper/`
2. **Test installation**:
   ```bash
   pip install smart-image-cropper
   python -c "from smart_image_cropper import SmartImageCropper; print('Success!')"
   ```
3. **Test in a fresh environment** to ensure all dependencies are properly
   specified

## Best Practices

1. **Always test on TestPyPI first**
2. **Use semantic versioning** (MAJOR.MINOR.PATCH)
3. **Keep detailed changelog** in README.md
4. **Use API tokens** instead of passwords
5. **Automate with CI/CD** for consistent releases
6. **Tag your releases** in Git:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

## Resources

- [PyPI Official Packaging Tutorial](https://packaging.python.org/tutorials/packaging-projects/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Setuptools Documentation](https://setuptools.readthedocs.io/)
- [Python Packaging User Guide](https://packaging.python.org/)

---

🎉 **Congratulations!** Your Smart Image Cropper library is now available for
the world to use via `pip install smart-image-cropper`!
