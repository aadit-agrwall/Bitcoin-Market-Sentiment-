import os
import pandas as pd
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), 'outputs')


def load_merged():
    path = os.path.join(OUT, 'merged_daily.csv')
    if not os.path.exists(path):
        raise FileNotFoundError('merged_daily.csv not found; run main.py first')
    return pd.read_csv(path, parse_dates=['date']).sort_values('date')


def simple_backtest(df, cluster_days=None, entry='open', exit='close'):
    # For each day, use total_closed_pnl as realized pnl (proxy). We'll simulate a strategy:
    # - If date in cluster_days, keep trades; else skip (zero pnl)
    res = df[['date', 'total_closed_pnl']].copy()
    res['keep'] = res['date'].isin(cluster_days) if cluster_days is not None else True
    res['strategy_pnl'] = res['total_closed_pnl'] * res['keep']
    res['cum_pnl'] = res['strategy_pnl'].cumsum()
    return res


def evaluate(df, label='strategy'):
    total = df['strategy_pnl'].sum()
    days = len(df)
    kept = int(df['keep'].sum())
    sharpe = (df['strategy_pnl'].mean() / (df['strategy_pnl'].std() if df['strategy_pnl'].std() != 0 else 1)) * np.sqrt(252)
    return {
        'label': label,
        'total_pnl': float(total),
        'days': int(days),
        'kept_days': int(kept),
        'sharpe_approx': float(sharpe)
    }


def main():
    merged = load_merged()
    clusters_path = os.path.join(OUT, 'merged_daily_clusters.csv')
    if not os.path.exists(clusters_path):
        raise FileNotFoundError('merged_daily_clusters.csv not found; run clustering script first')
    clusters = pd.read_csv(clusters_path, parse_dates=['date'])

    df = pd.merge(merged, clusters[['date', 'cluster']], on='date', how='left')

    # Baseline: keep all days
    baseline = simple_backtest(df, cluster_days=None)
    baseline['keep'] = True
    base_eval = evaluate(baseline, 'baseline')

    # Strategy: keep only days in cluster 1
    cluster1_days = df[df['cluster'] == 1]['date']
    strat = simple_backtest(df, cluster_days=cluster1_days)
    strat_eval = evaluate(strat, 'cluster_1_only')

    # Strategy: keep only days in cluster 0
    cluster0_days = df[df['cluster'] == 0]['date']
    strat0 = simple_backtest(df, cluster_days=cluster0_days)
    strat0_eval = evaluate(strat0, 'cluster_0_only')

    results = pd.DataFrame([base_eval, strat_eval, strat0_eval])
    results.to_csv(os.path.join(OUT, 'backtest_cluster_results.csv'), index=False)
    merged_out = os.path.join(OUT, 'backtest_cluster_timeseries.csv')
    pd.concat([baseline[['date','cum_pnl']].rename(columns={'cum_pnl':'baseline_cum_pnl'}),
               strat[['date','cum_pnl']].rename(columns={'cum_pnl':'cluster1_cum_pnl'})['cluster1_cum_pnl']], axis=1).to_csv(merged_out, index=False)

    print('Backtest results saved to', OUT)


if __name__ == '__main__':
    main()
