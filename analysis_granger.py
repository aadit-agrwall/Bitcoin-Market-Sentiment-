import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats


def compute_lagged_correlations(df, sent_col='value', pnl_col='total_closed_pnl', max_lag=30):
    s = df[sent_col]
    p = df[pnl_col]
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        s_shift = s.shift(lag)
        mask = s_shift.notna() & p.notna()
        n = int(mask.sum())
        if n >= 3:
            try:
                r, pval = stats.pearsonr(s_shift[mask], p[mask])
            except Exception:
                r, pval = np.nan, np.nan
        else:
            r, pval = np.nan, np.nan
        rows.append({'lag': lag, 'n': n, 'r': r, 'p': pval})
    return pd.DataFrame(rows)


def run_granger(df, x_col, y_col, maxlag=14):
    # Test whether x causes y (y first, x second as required by statsmodels)
    data = df[[y_col, x_col]].dropna()
    if data.shape[0] < 10:
        return None
    maxlag = min(maxlag, max(1, int(data.shape[0] / 2) - 1))
    res = grangercausalitytests(data[[y_col, x_col]].values, maxlag=maxlag, verbose=False)
    rows = []
    for lag, item in res.items():
        test_res = item[0]
        rows.append({
            'lag': lag,
            'ssr_ftest_p': float(test_res['ssr_ftest'][1]),
            'ssr_chi2test_p': float(test_res['ssr_chi2test'][1]),
            'lrtest_p': float(test_res['lrtest'][1]),
            'params_ftest_p': float(test_res['params_ftest'][1]),
        })
    return pd.DataFrame(rows).sort_values('lag')


def main():
    base = os.path.dirname(__file__)
    out = os.path.join(base, 'outputs')
    merged_path = os.path.join(out, 'merged_daily.csv')
    if not os.path.exists(merged_path):
        print('Error: merged_daily.csv not found. Run main.py first to generate it.')
        return

    df = pd.read_csv(merged_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    sent_col = 'value'
    pnl_col = 'total_closed_pnl'

    print('Computing lagged correlations...')
    lag_df = compute_lagged_correlations(df, sent_col=sent_col, pnl_col=pnl_col, max_lag=30)
    lag_df.to_csv(os.path.join(out, 'lagged_correlations.csv'), index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(lag_df['lag'], lag_df['r'], marker='o')
    plt.axhline(0, color='k', linewidth=0.6)
    plt.xlabel('Lag (positive: sentiment leads PnL)')
    plt.ylabel('Pearson r')
    plt.title('Lagged correlation between sentiment value and total_closed_pnl')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'lagged_correlations_plot.png'), dpi=150)
    plt.close()

    maxlag = 14
    print('Running Granger causality tests (value -> pnl)...')
    g_v_to_p = run_granger(df, x_col=sent_col, y_col=pnl_col, maxlag=maxlag)
    if g_v_to_p is not None:
        g_v_to_p.to_csv(os.path.join(out, 'granger_value_causes_pnl.csv'), index=False)

    print('Running Granger causality tests (pnl -> value)...')
    g_p_to_v = run_granger(df, x_col=pnl_col, y_col=sent_col, maxlag=maxlag)
    if g_p_to_v is not None:
        g_p_to_v.to_csv(os.path.join(out, 'granger_pnl_causes_value.csv'), index=False)

    def summarize(tbl):
        if tbl is None or tbl.empty:
            return 'not enough data'
        return f"{(tbl['ssr_ftest_p'] < 0.05).sum()} of {len(tbl)} lags significant (ssr_ftest p<0.05)"

    summary = {
        'lagged_correlations': 'lagged_correlations.csv',
        'lagged_plot': 'lagged_correlations_plot.png',
        'granger_value_causes_pnl': summarize(g_v_to_p),
        'granger_pnl_causes_value': summarize(g_p_to_v),
    }

    with open(os.path.join(out, 'granger_summary.txt'), 'w') as fh:
        fh.write('Granger / lagged-correlation summary\n\n')
        for k, v in summary.items():
            fh.write(f"{k}: {v}\n")

    print('Saved lagged correlations and Granger results to', out)


if __name__ == '__main__':
    main()
