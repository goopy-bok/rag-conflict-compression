#!/usr/bin/env bash
# OpenRouter Phi-4 isolated experiment environment.
# Usage: source scripts/env_openrouter.sh

export CCD_MODELS_CONFIG="configs/models_openrouter.yaml"
export CCD_CONDITIONS_CONFIG="configs/conditions_openrouter.yaml"
export CCD_RESULTS_DIR="results/openrouter_phi4"
export CCD_CACHE_DIR="cache_openrouter"
export CCD_PROMPTS_DIR="prompts/openrouter"
export CCD_REPORTS_DIR="reports/openrouter"

mkdir -p "$CCD_RESULTS_DIR" "$CCD_CACHE_DIR" "$CCD_PROMPTS_DIR" "$CCD_REPORTS_DIR"

echo "[CCD-G1] OpenRouter Phi-4 environment active."
echo "  Models config:     $CCD_MODELS_CONFIG"
echo "  Conditions config: $CCD_CONDITIONS_CONFIG"
echo "  Results directory: $CCD_RESULTS_DIR"
echo "  Cache directory:   $CCD_CACHE_DIR"
echo "  Prompts directory: $CCD_PROMPTS_DIR"
echo "  Reports directory: $CCD_REPORTS_DIR"
