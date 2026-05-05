#!/usr/bin/env bash
# deploy_docs.sh — Build Sphinx docs and push to gh-pages branch
# Usage: ./deploy_docs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_ROOT/docs/_build/html"

echo "==> Building Sphinx documentation..."
cd "$REPO_ROOT"
rm -rf "$BUILD_DIR"
sphinx-build -b html docs "$BUILD_DIR"

echo "==> Deploying to gh-pages branch..."
cd "$BUILD_DIR"
touch .nojekyll

# Init a temporary git repo in the build dir, commit, and force-push
git init -b gh-pages
git add -A
git commit -m "Deploy docs $(date -u '+%Y-%m-%d %H:%M UTC')"
git remote add origin "$(cd "$REPO_ROOT" && git remote get-url origin)"
git push --force origin gh-pages

# Clean up the temporary .git inside build dir
rm -rf "$BUILD_DIR/.git"

echo "==> Done! Docs deployed to gh-pages."
