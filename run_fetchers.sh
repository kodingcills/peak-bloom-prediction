#!/usr/bin/env bash
set -euo pipefail

# Phase 1 data fetchers — run this from YOUR terminal (not Claude Code)
# Requires: venv activated, network access

echo "=== Phase 1 Data Fetch ==="
echo "Started: $(date -u)"
echo ""

# Step 1: ERA5-Land (longest — 10-30 min)
echo ">>> ERA5-Land fetch..."
python3 refresh_data.py --step era5
echo ""

# Step 2: ASOS
echo ">>> ASOS fetch..."
python3 refresh_data.py --step asos
echo ""

# Step 3: SEAS5 (fallback mode — skip actual CDS fetch for now)
echo ">>> SEAS5 fetch (fallback mode)..."
SEAS5_FALLBACK_MODE=true python3 refresh_data.py --step seas5
echo ""

# Step 4: Gold features
echo ">>> Building gold features..."
python3 refresh_data.py --step features
echo ""

# Step 5: Verification
echo "=== Artifact Verification ==="
echo "Silver weather files:"
find data/silver/weather -name "*_consolidated.parquet" -exec ls -lh {} \;
echo ""
echo "ASOS files:"
find data/silver/asos -name "*.parquet" -exec ls -lh {} \; 2>/dev/null || echo "  (none)"
echo ""
echo "Gold features:"
ls -lh data/gold/features.parquet 2>/dev/null || echo "  (missing!)"
echo ""

# Step 6: Quick diagnostics
python3 -c "
import pandas as pd
df = pd.read_parquet('data/gold/features.parquet')
print('Gold features shape:', df.shape)
print('Columns:', list(df.columns))
print()
print('Year range per site:')
print(df.groupby('site_key')['year'].agg(['min', 'max', 'count']).to_string())
print()
print('2026 rows:')
print(df[df['year'] == 2026].to_string())
print()
print('GDH range:', df['gdh'].min(), '-', df['gdh'].max())
print('CP range:', df['cp'].min(), '-', df['cp'].max())
print('Null bloom_doy count:', df['bloom_doy'].isna().sum(), '(should be 5)')
"

echo ""
echo "=== Fetch complete: $(date -u) ==="
echo "Now return to Claude Code and run: python3 -m src.validation.run_all_gates --phase 1"
