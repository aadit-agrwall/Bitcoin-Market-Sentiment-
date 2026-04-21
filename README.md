Project: Sentiment vs Trader Performance

This script cleans and analyzes two datasets placed in the repository root:
- `fear_greed_index (1).csv` — Bitcoin Fear & Greed index
- `historical_data (1).csv` — trader historical execution data

Run:

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

Outputs will be written to the `outputs/` folder.

## Release Summary (initial)

- Initial commit includes ETL and analysis scripts, clustering and backtest code, and an interactive notebook for exploring lag relationships.
- Key files:
	- `main.py` — data cleaning, aggregation, summary statistics, and static plots.
	- `analysis_granger.py` — lagged correlations and Granger causality tests.
	- `analysis_accounts_clusters.py` — per-account/symbol aggregation and clustering.
	- `backtest_clusters.py` — simple cluster-based backtest using `total_closed_pnl` as PnL proxy.
	- `explore_clusters_lags.ipynb` — interactive notebook (ipywidgets) to explore windows and lags.
- Environment: Create and use the provided virtualenv (`venv`) before running scripts; install `ipywidgets` for the notebook: `./venv/bin/python -m pip install ipywidgets`.

## How I ran and pushed this repo

- I initialized a local git repo, added a sensible `.gitignore`, committed the repository, and pushed the `main` branch to:

	https://github.com/aadit-agrwall/Bitcoin-Market-Sentiment-.git

If you'd like, I can add CI (GitHub Actions) to run basic checks or a short release note file. Open an issue or tell me what to include next.

## CI / Checks

- A GitHub Actions workflow `CI` has been added to run on pushes and pull requests to `main`.
- The workflow installs pinned dependencies from `requirements.txt`, runs `black --check` and `flake8`, and performs a smoke-run of `main.py` to catch runtime errors.

## License

This project is provided under the MIT License — see the `LICENSE` file.
