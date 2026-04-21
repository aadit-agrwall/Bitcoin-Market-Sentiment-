import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


DATA_DIR = os.path.dirname(__file__)
HIST_FILE = os.path.join(DATA_DIR, 'historical_data (1).csv')
FG_FILE = os.path.join(DATA_DIR, 'fear_greed_index (1).csv')
MERGED_DAILY = os.path.join(DATA_DIR, 'outputs', 'merged_daily.csv')
OUT_DIR = os.path.join(DATA_DIR, 'outputs')


def load_trades(path=HIST_FILE):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # parse human-readable IST timestamp (day-first)
    if 'Timestamp IST' in df.columns:
        df['ts'] = pd.to_datetime(df['Timestamp IST'], dayfirst=True, errors='coerce')
    elif 'Timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['Timestamp'], unit='s', errors='coerce')
    else:
        df['ts'] = pd.NaT

    df['date'] = pd.to_datetime(df['ts']).dt.floor('D')

    # numeric conversions
    for col in ['Closed PnL', 'Size USD', 'Execution Price', 'Fee']:
        if col in df.columns:
            # remove commas and coerce
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # normalize account and coin column names
    if 'Account' not in df.columns and 'account' in df.columns:
        df['Account'] = df['account']
    if 'Coin' not in df.columns and 'symbol' in df.columns:
        df['Coin'] = df['symbol']

    return df


def load_sentiment(path=FG_FILE):
    fg = pd.read_csv(path)
    if 'date' in fg.columns:
        fg['date'] = pd.to_datetime(fg['date']).dt.floor('D')
    elif 'timestamp' in fg.columns:
        fg['date'] = pd.to_datetime(pd.to_numeric(fg['timestamp'], errors='coerce'), unit='s').dt.floor('D')
    return fg


def per_account_symbol_analysis(trades, fg):
    # merge sentiment onto trades by date
    df = trades.copy()
    fg2 = fg[['date', 'value', 'classification']].drop_duplicates()
    df = pd.merge(df, fg2, on='date', how='left')

    # per-account by sentiment
    group_cols = ['Account', 'classification']
    agg = df.groupby(group_cols).agg(
        trades_count=('Account', 'size'),
        total_closed_pnl=('Closed PnL', 'sum'),
        avg_closed_pnl=('Closed PnL', 'mean'),
        win_rate=('Closed PnL', lambda g: (g > 0).sum() / g.count() if g.count() > 0 else 0),
        total_volume=('Size USD', 'sum'),
        avg_volume=('Size USD', 'mean')
    ).reset_index()
    agg.to_csv(os.path.join(OUT_DIR, 'per_account_by_sentiment.csv'), index=False)

    # per-symbol by sentiment
    if 'Coin' in df.columns:
        grp_sym = df.groupby(['Coin', 'classification']).agg(
            trades_count=('Coin', 'size'),
            total_closed_pnl=('Closed PnL', 'sum'),
            avg_closed_pnl=('Closed PnL', 'mean'),
            win_rate=('Closed PnL', lambda g: (g > 0).sum() / g.count() if g.count() > 0 else 0),
            total_volume=('Size USD', 'sum')
        ).reset_index()
        grp_sym.to_csv(os.path.join(OUT_DIR, 'per_symbol_by_sentiment.csv'), index=False)
    else:
        grp_sym = None

    # top accounts by total pnl overall
    acc_overall = df.groupby('Account').agg(total_pnl=('Closed PnL', 'sum'), trades=('Account', 'size')).reset_index()
    acc_overall = acc_overall.sort_values('trades', ascending=False)
    acc_overall.to_csv(os.path.join(OUT_DIR, 'per_account_overall.csv'), index=False)

    # plots for top accounts and top symbols
    top_accounts = acc_overall.head(6)['Account'].tolist()
    if len(top_accounts) > 0:
        pivot = agg[agg['Account'].isin(top_accounts)].pivot(index='Account', columns='classification', values='avg_closed_pnl')
        pivot = pivot.reindex(top_accounts)
        plt.figure(figsize=(10, 5))
        pivot.plot(kind='bar', rot=45)
        plt.ylabel('Avg Closed PnL')
        plt.title('Top accounts: Avg Closed PnL by Sentiment')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'top_accounts_avg_pnl_by_sentiment.png'), dpi=150)
        plt.close()

    if grp_sym is not None:
        top_symbols = df['Coin'].value_counts().head(8).index.tolist()
        pivot_s = grp_sym[grp_sym['Coin'].isin(top_symbols)].pivot(index='Coin', columns='classification', values='avg_closed_pnl')
        plt.figure(figsize=(10, 5))
        pivot_s.plot(kind='bar', rot=45)
        plt.ylabel('Avg Closed PnL')
        plt.title('Top symbols: Avg Closed PnL by Sentiment')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'top_symbols_avg_pnl_by_sentiment.png'), dpi=150)
        plt.close()

    return agg, grp_sym


def cluster_trading_days(merged_daily_path=MERGED_DAILY, min_k=2, max_k=6):
    if not os.path.exists(merged_daily_path):
        print('Merged daily data not found at', merged_daily_path)
        return None

    df = pd.read_csv(merged_daily_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # choose features
    features = []
    for c in ['value', 'total_closed_pnl', 'avg_closed_pnl', 'trades_count', 'win_rate']:
        if c in df.columns:
            features.append(c)

    X = df[features].fillna(0).copy()
    # reduce skew for pnl columns
    for col in ['total_closed_pnl', 'avg_closed_pnl']:
        if col in X.columns:
            X[col] = np.sign(X[col]) * np.log1p(np.abs(X[col]))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best_k = None
    best_score = -1
    best_labels = None
    best_model = None
    for k in range(min_k, min(max_k + 1, max(3, len(df) - 1))):
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init='auto')
            labels = km.fit_predict(Xs)
            if len(set(labels)) > 1:
                score = silhouette_score(Xs, labels)
            else:
                score = -1
        except Exception:
            score = -1
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_model = km

    df['cluster'] = best_labels if best_labels is not None else 0
    df.to_csv(os.path.join(OUT_DIR, 'merged_daily_clusters.csv'), index=False)

    # PCA for visualization
    pca = PCA(2)
    Xp = pca.fit_transform(Xs)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Xp[:, 0], y=Xp[:, 1], hue=df['cluster'], palette='tab10', legend='full')
    plt.title(f'PCA projection colored by cluster (k={best_k}, silhouette={best_score:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'clusters_pca.png'), dpi=150)
    plt.close()

    # cluster timeline
    plt.figure(figsize=(12, 2))
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in df['cluster']]
    plt.scatter(df['date'], [1] * len(df), c=df['cluster'], cmap='tab10', marker='s', s=20)
    plt.yticks([])
    plt.title('Cluster assignment over time')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cluster_timeline.png'), dpi=150)
    plt.close()

    # summary
    cluster_summary = df.groupby('cluster')[features].mean()
    cluster_summary.to_csv(os.path.join(OUT_DIR, 'cluster_feature_summary.csv'))

    with open(os.path.join(OUT_DIR, 'clustering_summary.txt'), 'w') as fh:
        fh.write(f'Best k: {best_k}\n')
        fh.write(f'Silhouette score: {best_score}\n')
        fh.write('\nCluster centers (features means):\n')
        fh.write(cluster_summary.to_string())

    return df, best_k, best_score


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading trades and sentiment...')
    trades = load_trades()
    fg = load_sentiment()

    print('Computing per-account and per-symbol summaries...')
    per_acc, per_sym = per_account_symbol_analysis(trades, fg)
    print('Saved per-account and per-symbol CSVs and plots to', OUT_DIR)

    print('Clustering trading days...')
    df_clusters, best_k, best_score = cluster_trading_days()
    print('Clustering done. Best k=', best_k, 'silhouette=', best_score)


if __name__ == '__main__':
    main()
