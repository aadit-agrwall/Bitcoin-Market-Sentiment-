
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.dirname(__file__)
FG_FILE = os.path.join(DATA_DIR, 'fear_greed_index (1).csv')
HIST_FILE = os.path.join(DATA_DIR, 'historical_data (1).csv')
OUT_DIR = os.path.join(DATA_DIR, 'outputs')


def ensure_output_dir():
	os.makedirs(OUT_DIR, exist_ok=True)


def load_fear_greed(path=FG_FILE):
	df = pd.read_csv(path)
	# timestamps in file are UNIX seconds; keep both parsed ts and date
	if 'timestamp' in df.columns:
		df['ts'] = pd.to_datetime(pd.to_numeric(df['timestamp'], errors='coerce'), unit='s', errors='coerce')
	else:
		df['ts'] = pd.to_datetime(df.get('date'), errors='coerce')

	# ensure a proper date column (floor to day)
	if 'date' in df.columns:
		df['date'] = pd.to_datetime(df['date'], errors='coerce')
	else:
		df['date'] = df['ts'].dt.floor('D')

	df['value'] = pd.to_numeric(df['value'], errors='coerce')
	df['classification'] = df['classification'].astype(str)
	df = df[['ts', 'date', 'value', 'classification']].dropna(subset=['date'])
	return df


def load_historical(path=HIST_FILE):
	df = pd.read_csv(path)
	# normalize column names
	df.columns = [c.strip() for c in df.columns]

	# parse human-readable IST timestamp if present (assumes day-first format)
	if 'Timestamp IST' in df.columns:
		df['ts_ist'] = pd.to_datetime(df['Timestamp IST'], dayfirst=True, errors='coerce')
	else:
		df['ts_ist'] = pd.NaT

	# parse numeric epoch timestamp if present (scientific notation -> ms)
	if 'Timestamp' in df.columns:
		df['ts_epoch'] = pd.to_numeric(df['Timestamp'], errors='coerce')
		df.loc[df['ts_epoch'] > 1e11, 'ts_epoch_dt'] = pd.to_datetime(df.loc[df['ts_epoch'] > 1e11, 'ts_epoch'].astype('int64'), unit='ms', errors='coerce')
		df.loc[df['ts_epoch'] <= 1e11, 'ts_epoch_dt'] = pd.to_datetime(df.loc[df['ts_epoch'] <= 1e11, 'ts_epoch'].astype('int64'), unit='s', errors='coerce')
	else:
		df['ts_epoch_dt'] = pd.NaT

	# choose the best available timestamp
	df['ts'] = df['ts_ist'].fillna(df['ts_epoch_dt'])
	df['date'] = pd.to_datetime(df['ts']).dt.floor('D')

	# numeric conversions
	for col in ['Closed PnL', 'Size USD', 'Execution Price', 'Fee']:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

	# normalize direction/side
	if 'Direction' in df.columns:
		df['direction'] = df['Direction'].astype(str)
	else:
		df['direction'] = df.get('Side', '').astype(str)

	return df


def aggregate_daily(df):
	# focus on realized/closed trades where Closed PnL != 0
	df_closed = df[df['Closed PnL'].notnull() & (df['Closed PnL'] != 0)]

	daily_all = df.groupby('date').agg(trades_count=('Account', 'count'), total_volume=('Size USD', 'sum'))
	if not df_closed.empty:
		daily_closed = df_closed.groupby('date').agg(closed_trades=('Account', 'count'), total_closed_pnl=('Closed PnL', 'sum'), avg_closed_pnl=('Closed PnL', 'mean'), median_closed_pnl=('Closed PnL', 'median'))
		win_rate = df_closed.groupby('date').apply(lambda g: (g['Closed PnL'] > 0).sum() / len(g))
		daily = daily_all.join(daily_closed, how='left')
		daily['win_rate'] = win_rate
	else:
		daily = daily_all.copy()
		daily['closed_trades'] = 0
		daily['total_closed_pnl'] = 0.0
		daily['avg_closed_pnl'] = 0.0
		daily['median_closed_pnl'] = 0.0
		daily['win_rate'] = 0.0

	daily = daily.fillna(0).reset_index()
	return daily


def merge_with_sentiment(daily, fg):
	fg2 = fg.copy()
	fg2['date'] = pd.to_datetime(fg2['date']).dt.floor('D')
	merged = pd.merge(daily, fg2[['date', 'value', 'classification']], on='date', how='left')
	return merged


def analyze_and_save(merged):
	out_csv = os.path.join(OUT_DIR, 'merged_daily.csv')
	merged.to_csv(out_csv, index=False)

	# summary by sentiment classification
	group = merged.groupby('classification').agg(days=('date', 'count'), avg_closed_pnl=('avg_closed_pnl', 'mean'), total_closed_pnl=('total_closed_pnl', 'sum'), avg_win_rate=('win_rate', 'mean'))
	group.to_csv(os.path.join(OUT_DIR, 'summary_by_sentiment.csv'))

	# simple correlation
	corr = np.nan
	try:
		tmp = merged[['value', 'total_closed_pnl']].dropna()
		if tmp.shape[0] > 1:
			corr = tmp['value'].corr(tmp['total_closed_pnl'])
	except Exception:
		corr = np.nan

	with open(os.path.join(OUT_DIR, 'analysis_summary.txt'), 'w') as f:
		f.write(f'Correlation between sentiment value and total_closed_pnl: {corr}\n')
		f.write('\nSummary by sentiment classification:\n')
		f.write(group.to_string())

	print('Saved merged and summary files to', OUT_DIR)
	return corr, group


def plot(merged):
	sns.set(style='whitegrid')
	merged_sorted = merged.sort_values('date')

	# time series: sentiment value and total_closed_pnl
	plt.figure(figsize=(12, 6))
	ax1 = plt.gca()
	if 'value' in merged_sorted.columns:
		ax1.plot(merged_sorted['date'], merged_sorted['value'], color='tab:blue', label='Sentiment value')
	ax2 = ax1.twinx()
	if 'total_closed_pnl' in merged_sorted.columns:
		ax2.plot(merged_sorted['date'], merged_sorted['total_closed_pnl'], color='tab:orange', label='Total Closed PnL')
	ax1.set_xlabel('Date')
	ax1.set_ylabel('Sentiment value')
	ax2.set_ylabel('Total Closed PnL')
	plt.title('Sentiment (value) and Total Closed PnL over time')
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, 'sentiment_vs_pnl_timeseries.png'), dpi=150)
	plt.close()

	# boxplot of avg_closed_pnl by sentiment classification
	plt.figure(figsize=(8, 6))
	order = sorted(merged['classification'].dropna().unique())
	sns.boxplot(x='classification', y='avg_closed_pnl', data=merged, order=order)
	plt.xticks(rotation=45)
	plt.title('Avg Closed PnL by Sentiment Classification')
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, 'boxplot_pnl_by_sentiment.png'), dpi=150)
	plt.close()

	# scatter sentiment value vs total_closed_pnl
	plt.figure(figsize=(8, 6))
	sns.scatterplot(data=merged, x='value', y='total_closed_pnl')
	try:
		sns.regplot(data=merged.dropna(subset=['value', 'total_closed_pnl']), x='value', y='total_closed_pnl', scatter=False, color='red')
	except Exception:
		pass
	plt.title('Sentiment value vs Total Closed PnL')
	plt.tight_layout()
	plt.savefig(os.path.join(OUT_DIR, 'scatter_value_vs_pnl.png'), dpi=150)
	plt.close()

	print('Saved plots to', OUT_DIR)


def main():
	ensure_output_dir()
	print('Loading datasets...')
	fg = load_fear_greed()
	print('Sentiment date range:', fg['date'].min(), 'to', fg['date'].max())

	hist = load_historical()
	print('Trader data date range:', hist['date'].min(), 'to', hist['date'].max())

	print('Aggregating trader data by day...')
	daily = aggregate_daily(hist)

	print('Merging with sentiment...')
	merged = merge_with_sentiment(daily, fg)

	print('Analyzing and saving results...')
	corr, group = analyze_and_save(merged)
	print('Correlation (value vs total_closed_pnl):', corr)

	print('Generating plots...')
	plot(merged)
	print('Done. Check the outputs/ folder for CSVs, plots, and analysis summary.')


if __name__ == '__main__':
	main()
