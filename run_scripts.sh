#!/bin/bash
# This script will run the Python files in the correct order

echo "Running cartography.py"
python3 /app/cartographic_analysis/cartography.py

echo "Running noise_a_bias.py"
python3 /app/cartographic_analysis/noise_a_bias.py

echo "Running applied.py"
python3 /app/cartographic_analysis/applied.py

echo "All scripts executed successfully!"
