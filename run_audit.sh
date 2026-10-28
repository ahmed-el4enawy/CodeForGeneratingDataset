#!/bin/bash
#SBATCH --job-name=val_audit
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=audit_%j.out
#SBATCH --error=audit_%j.err

set -euo pipefail

echo "=========================================="
echo "Starting Valentina Dataset Audit Job"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Select Python environment: use active virtualenv or default HPC env
if command -v python &> /dev/null; then
    PYTHON_BIN="python"
elif [ -f "/nfs/slurm/cugp012/envs/t1d/bin/python" ]; then
    PYTHON_BIN="/nfs/slurm/cugp012/envs/t1d/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

OUTPUT_DIR="audit_results"
mkdir -p "$OUTPUT_DIR"

echo "------------------------------------------"
echo "1. Running Full Population Dataset Audit..."
echo "------------------------------------------"
$PYTHON_BIN audit_population_dataset.py \
    --root GeneratedDataset/OHSUSimPopulation \
    --output-dir "$OUTPUT_DIR"

echo "------------------------------------------"
echo "2. Running Clinical Metrics Audit (SIM002/TEST)..."
echo "------------------------------------------"
$PYTHON_BIN audit_valentina_test_metrics.py \
    --test-dir GeneratedDataset/OHSUSimPopulation/SIM002/POPULATION/TEST \
    --output-dir "$OUTPUT_DIR"

echo "=========================================="
echo "Audit Completed Successfully: $(date)"
echo "Output files located in: $OUTPUT_DIR"
echo "=========================================="
ls -lh "$OUTPUT_DIR"
