Place raw two-step CSVs here.

Recommended structure:
- `two_step_task/data/raw/hartley_twostep_data.csv`  # raw CSV from OSF
- `two_step_task/data/processed/`                    # per-participant processed files

Quick inspection script:
- Use `two_step_task/scripts/inspect_csv_format.py` to list CSV files and print columns/sample rows.
	Example (Windows cmd from repo root):

```cmd
.venv\\Scripts\\python.exe two_step_task\\scripts\\inspect_csv_format.py --path two_step_task\\data\\raw
```

This script will report missing required columns and show column dtypes and the first 5 rows of each CSV.
