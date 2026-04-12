#!/usr/bin/env bash
# ╔═══════════════════════════════════════════════════════════════╗
# ║  CartoCrypt — Repository initialisation script                ║
# ║  « Run once after cloning or extracting the scaffold »        ║
# ╚═══════════════════════════════════════════════════════════════╝
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "── Initialising git repository ──────────────────────────"
git init
git add .
git commit -m "feat: initial CartoCrypt scaffold

- Package structure: src/cartocrypt with 10 modules
- CLI via Click (keygen, anonymise, verify)
- pyproject.toml with setuptools-scm versioning
- Conda environment.yml (Python 3.11)
- GitHub Actions CI (test matrix, docs, release)
- Sphinx RTD-theme docs with autodoc
- CITATION.cff for Zenodo DOI
- MIT LICENSE
- pytest suite for keygen, canon, reembed"

echo ""
echo "── Setting up remote ────────────────────────────────────"
echo "Run:"
echo "  git remote add origin git@github.com:zerotonin/cartocrypt.git"
echo "  git push -u origin main"
echo ""
echo "── Installing in dev mode ───────────────────────────────"
echo "Run:"
echo "  conda env create -f environment.yml"
echo "  conda activate cartocrypt"
echo "  pip install -e '.[dev]' --break-system-packages"
echo ""
echo "── Running tests ────────────────────────────────────────"
echo "Run:"
echo "  pytest"
echo ""
echo "Done."
