"""Generates SMA_DCA_BnHTest.ipynb as a clean standalone notebook."""
import json, uuid

def cid(): return str(uuid.uuid4())[:8]

def md(src):
    lines = src.split('\n')
    source = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {'cell_type': 'markdown', 'id': cid(), 'metadata': {}, 'source': source}

def code(src):
    lines = src.strip('\n').split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {'cell_type': 'code', 'id': cid(), 'metadata': {},
            'source': source, 'outputs': [], 'execution_count': None}

cells = []

# ── Cell 0: Title ─────────────────────────────────────────────────────────────
cells.append(md("""\
# SMA 5 / SMA 100 Crossover vs Simple DCA vs Buy & Hold
## Last 2 BTC Halving Cycles  (May 2020 → present)

**Goal:** Determine whether the SMA 5/100 crossover signal outperforms passive strategies
over the last two complete Bitcoin halving cycles.

| Strategy | Logic |
|----------|-------|
| **SMA 5/100 Crossover** | All-in long when SMA-5 > SMA-100; exit to cash on cross-under |
| **Simple DCA** | Invest `$10,000 ÷ n_trading_days` every day (same total capital as B&H) |
| **Buy & Hold** | Full $10,000 on day 1 of Cycle 3, hold until today |

**Parameters:** $10,000 starting capital · 0.1% fee per trade · RF = 4%
**Cycle boundary:** 4th BTC halving — April 19 2024 (dashed vertical line on charts)"""))

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
cells.append(code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import yfinance as yf
from datetime import datetime
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
print('Libraries loaded.')\
"""))

# ── Cell 2: Fetch BTC daily data ──────────────────────────────────────────────
cells.append(code("""\
def fetch_btc(start='2018-01-01'):
    raw = yf.Ticker('BTC-USD').history(start=start, interval='1d')
    raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
    raw.rename(columns={'Date':'date','Open':'open','High':'high',
                        'Low':'low','Close':'close','Volume':'volume'}, inplace=True)
    # Strip any timezone info from the date column
    raw['date'] = pd.to_datetime(raw['date'])
    if raw['date'].dt.tz is not None:
        raw['date'] = raw['date'].dt.tz_localize(None)
    raw['date'] = raw['date'].dt.normalize()
    return raw.sort_values('date').reset_index(drop=True)

data    = fetch_btc(start='2018-01-01')
price_s = pd.Series(data['close'].values,
                    index=pd.to_datetime(data['date']),
                    name='close')

print(f'BTC daily: {len(data)} bars  |  {price_s.index[0].date()} -> {price_s.index[-1].date()}')
data.tail(3)\
"""))

# ── Cell 3: Compute SMAs + run all 3 strategies ───────────────────────────────
cells.append(code("""\
CYCLE_3_START = '2020-05-11'    # 3rd BTC halving
CYCLE_4_START = '2024-04-19'    # 4th BTC halving (cycle boundary)
CYCLE_CAPITAL = 10_000
FEE           = 0.001

# ── Slice window ──────────────────────────────────────────────────────────────
cycle_price = price_s.loc[CYCLE_3_START:].copy()
n_days      = len(cycle_price)
c4_mark     = pd.Timestamp(CYCLE_4_START)

print(f'Window : {cycle_price.index[0].date()} -> {cycle_price.index[-1].date()}  ({n_days} trading days)')
print(f'Cycle 3: {CYCLE_3_START} -> {CYCLE_4_START}  ({(c4_mark - cycle_price.index[0]).days} days)')
print(f'Cycle 4: {CYCLE_4_START} -> present  ({(cycle_price.index[-1] - c4_mark).days} days)')

# ── Compute SMAs on full history (warmup before cycle start = no look-ahead bias) ──
fast_ma_full = price_s.rolling(5).mean()
slow_ma_full = price_s.rolling(100).mean()
fast_c = fast_ma_full.reindex(cycle_price.index)
slow_c = slow_ma_full.reindex(cycle_price.index)

# ── Strategy 1: SMA 5/100 crossover ──────────────────────────────────────────
p_arr = cycle_price.values
f_arr = fast_c.values
s_arr = slow_c.values

pos = (f_arr > s_arr).astype(int)
chg = np.diff(pos, prepend=pos[0])

cash_sma = float(CYCLE_CAPITAL); btc_sma = 0.0
pv_sma_arr = np.empty(n_days); n_trades = 0
buys = []; sells = []

for i in range(n_days):
    if chg[i] == 1 and cash_sma > 0:
        btc_sma  = cash_sma * (1 - FEE) / p_arr[i]
        cash_sma = 0.0; n_trades += 1
        buys.append((cycle_price.index[i], p_arr[i]))
    elif chg[i] == -1 and btc_sma > 0:
        cash_sma = btc_sma * p_arr[i] * (1 - FEE)
        btc_sma  = 0.0; n_trades += 1
        sells.append((cycle_price.index[i], p_arr[i]))
    pv_sma_arr[i] = cash_sma + btc_sma * p_arr[i]

pv_sma = pd.Series(pv_sma_arr, index=cycle_price.index)

# ── Strategy 2: Simple DCA ────────────────────────────────────────────────────
daily_invest = CYCLE_CAPITAL / n_days
dca_cash = float(CYCLE_CAPITAL); dca_btc = 0.0
pv_dca_arr = []

for p in p_arr:
    if dca_cash > 0.01:
        spend     = min(daily_invest, dca_cash)
        dca_btc  += spend * (1 - FEE) / p
        dca_cash -= spend
    pv_dca_arr.append(dca_cash + dca_btc * p)

pv_dca = pd.Series(pv_dca_arr, index=cycle_price.index)

# ── Strategy 3: Buy & Hold ────────────────────────────────────────────────────
bh_btc = CYCLE_CAPITAL * (1 - FEE) / p_arr[0]
pv_bh  = pd.Series(bh_btc * p_arr, index=cycle_price.index)

STRAT_COLORS = {'sma': '#58a6ff', 'dca': '#3fb950', 'bh': '#e6b800'}

print(f'\\nSMA 5/100:  {n_trades} trades  ({len(buys)} buys / {len(sells)} sells)')
print(f'Simple DCA: ${daily_invest:.2f} / day for {n_days} days')
print()
for label, pv in [('SMA 5/100 ', pv_sma), ('Simple DCA', pv_dca), ('Buy & Hold', pv_bh)]:
    ret = (pv.iloc[-1] / CYCLE_CAPITAL - 1) * 100
    print(f'  {label}: ${pv.iloc[-1]:>10,.0f}  ({ret:+.1f}%)')\
"""))

# ── Cell 4: 3-panel comparison chart ─────────────────────────────────────────
cells.append(code("""\
today_str = datetime.today().strftime('%Y-%m-%d')
HALVING_C = '#f0883e'

fig, axes = plt.subplots(3, 1, figsize=(18, 20), facecolor='#0d1117',
                          gridspec_kw={'height_ratios': [2.5, 1.5, 3.0], 'hspace': 0.32})

# ── Panel 1: Equity curves ────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#161b22')
for pv, key, lbl in [(pv_sma, 'sma', 'SMA 5/100'),
                      (pv_dca, 'dca', 'Simple DCA'),
                      (pv_bh,  'bh',  'Buy & Hold')]:
    ret = (pv.iloc[-1] / CYCLE_CAPITAL - 1) * 100
    ax.plot(pv.index, pv.values, color=STRAT_COLORS[key], lw=1.8,
            label=f'{lbl}  ->  ${pv.iloc[-1]:,.0f}  ({ret:+.0f}%)')
ax.axvline(c4_mark, color=HALVING_C, lw=1.2, ls='--', alpha=0.8,
           label='4th Halving (Apr 2024)')
ax.axhline(CYCLE_CAPITAL, color='white', lw=0.5, ls=':', alpha=0.25)
y_top = max(pv_sma.max(), pv_dca.max(), pv_bh.max()) * 1.04
ax.text(pd.Timestamp('2022-01-01'), y_top * 0.98, 'Cycle 3',
        color=HALVING_C, fontsize=9, alpha=0.7, ha='center')
ax.text(pd.Timestamp('2025-06-01'), y_top * 0.98, 'Cycle 4',
        color=HALVING_C, fontsize=9, alpha=0.7, ha='center')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title(
    f'SMA 5/100  vs  Simple DCA  vs  Buy & Hold  |  Last 2 BTC Cycles  |  {today_str}\\n'
    f'$10,000 starting capital  *  0.1% fee/trade',
    color='white', fontsize=13, fontweight='bold', pad=10)
ax.legend(facecolor='#21262d', labelcolor='white', fontsize=10)
ax.grid(alpha=0.10); ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_color('#30363d')

# ── Panel 2: Drawdown ─────────────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor('#161b22')
for pv, key, lbl in [(pv_sma, 'sma', 'SMA 5/100'),
                      (pv_dca, 'dca', 'Simple DCA'),
                      (pv_bh,  'bh',  'Buy & Hold')]:
    dd = (pv - pv.cummax()) / pv.cummax() * 100
    ax.plot(dd.index, dd.values, color=STRAT_COLORS[key], lw=1.0, alpha=0.85, label=lbl)
    ax.fill_between(dd.index, dd.values, 0, color=STRAT_COLORS[key], alpha=0.06)
ax.axvline(c4_mark, color=HALVING_C, lw=1.2, ls='--', alpha=0.8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.set_title('Drawdown from Peak', color='white', fontsize=11, pad=6)
ax.legend(facecolor='#21262d', labelcolor='white', fontsize=9, ncol=3)
ax.grid(alpha=0.10); ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_color('#30363d')

# ── Panel 3: BTC price + SMAs + trade signals ─────────────────────────────────
ax = axes[2]
ax.set_facecolor('#161b22')
ax.plot(cycle_price.index, cycle_price.values,
        color='#444d56', lw=0.9, alpha=0.9, label='BTC Price')
ax.plot(fast_c.index, fast_c.values,
        color='#79c0ff', lw=1.2, alpha=0.85, label='SMA 5  (fast)')
ax.plot(slow_c.index, slow_c.values,
        color='#e6b800', lw=1.5, alpha=0.85, label='SMA 100  (slow)')
ax.fill_between(cycle_price.index, cycle_price.values, 0,
                where=(fast_c.values > slow_c.values),
                color='#3fb950', alpha=0.06, label='Long zone')
if buys:
    bd, bp = zip(*buys)
    ax.scatter(bd, bp, marker='^', color='#3fb950', s=90, zorder=5,
               label=f'Buy ({len(buys)})')
if sells:
    sd, sp = zip(*sells)
    ax.scatter(sd, sp, marker='v', color='#f85149', s=90, zorder=5,
               label=f'Sell ({len(sells)})')
ax.axvline(c4_mark, color=HALVING_C, lw=1.2, ls='--', alpha=0.8, label='4th Halving')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title(f'BTC Price  +  SMA 5 / SMA 100  +  Trade Signals  |  {n_trades} total trades',
             color='white', fontsize=11, pad=6)
ax.legend(facecolor='#21262d', labelcolor='white', fontsize=9, ncol=4)
ax.grid(alpha=0.10); ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_color('#30363d')

plt.savefig('sma5_100_cycle_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()\
"""))

# ── Cell 5: Metrics table + per-cycle bar chart ───────────────────────────────
cells.append(code("""\
def compute_metrics(pv, label, start_cap=None):
    base  = pv.iloc[0] if start_cap is None else start_cap
    final = pv.iloc[-1]
    total = (final / base - 1) * 100
    n_yrs = (pv.index[-1] - pv.index[0]).days / 365.25
    cagr  = ((final / base) ** (1 / n_yrs) - 1) * 100 if n_yrs > 0.01 else 0.0
    dr    = pv.pct_change().dropna()
    rf    = (1.04 ** (1 / 365.25)) - 1
    std   = dr.std()
    exc   = dr - rf
    shrp  = exc.mean() / std * np.sqrt(365.25) if std > 0 else 0.0
    down  = dr[dr < rf]
    srt   = (exc.mean() / down.std() * np.sqrt(365.25)
             if (len(down) > 1 and down.std() > 0) else 0.0)
    max_dd = ((pv - pv.cummax()) / pv.cummax() * 100).min()
    return {'Strategy': label,
            'Return (%)': f'{total:+.1f}', 'CAGR (%)': f'{cagr:.1f}',
            'Max DD (%)': f'{max_dd:.1f}', 'Sharpe': f'{shrp:.2f}',
            'Sortino': f'{srt:.2f}', 'Final ($)': f'${final:,.0f}'}

c4_idx = cycle_price.index.searchsorted(c4_mark)

# Full-window
rows_full = [
    {**compute_metrics(pv_sma, 'SMA 5/100',   CYCLE_CAPITAL), 'Trades': str(n_trades)},
    {**compute_metrics(pv_dca, 'Simple DCA',   CYCLE_CAPITAL), 'Trades': f'{n_days} (daily)'},
    {**compute_metrics(pv_bh,  'Buy & Hold',   CYCLE_CAPITAL), 'Trades': '1'},
]
df_full = (pd.DataFrame(rows_full)
             .set_index('Strategy')
             [['Trades','Return (%)','CAGR (%)','Max DD (%)','Sharpe','Sortino','Final ($)']])

print('=' * 72)
print(f'  FULL WINDOW  ({cycle_price.index[0].date()} -> {cycle_price.index[-1].date()})')
print('=' * 72)
display(df_full)

# Per-cycle sub-tables
rows_c3 = [compute_metrics(pv_sma.iloc[:c4_idx], 'SMA 5/100'),
           compute_metrics(pv_dca.iloc[:c4_idx], 'Simple DCA'),
           compute_metrics(pv_bh.iloc[:c4_idx],  'Buy & Hold')]
rows_c4 = [compute_metrics(pv_sma.iloc[c4_idx:], 'SMA 5/100'),
           compute_metrics(pv_dca.iloc[c4_idx:], 'Simple DCA'),
           compute_metrics(pv_bh.iloc[c4_idx:],  'Buy & Hold')]

cols = ['Return (%)','CAGR (%)','Max DD (%)','Sharpe','Sortino']
print(f'\\n  CYCLE 3  ({cycle_price.index[0].date()} -> {cycle_price.index[c4_idx].date()})')
display(pd.DataFrame(rows_c3).set_index('Strategy')[cols])

print(f'\\n  CYCLE 4  ({cycle_price.index[c4_idx].date()} -> {cycle_price.index[-1].date()})')
display(pd.DataFrame(rows_c4).set_index('Strategy')[cols])

# ── Per-cycle return bar chart ─────────────────────────────────────────────────
strat_names   = ['SMA 5/100', 'Simple DCA', 'Buy & Hold']
BAR_COLORS    = [STRAT_COLORS['sma'], STRAT_COLORS['dca'], STRAT_COLORS['bh']]
ret_c3_vals   = [float(r['Return (%)']) for r in rows_c3]
ret_c4_vals   = [float(r['Return (%)']) for r in rows_c4]
ret_full_vals = [(pv.iloc[-1] / CYCLE_CAPITAL - 1) * 100
                 for pv in [pv_sma, pv_dca, pv_bh]]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0d1117')
fig.suptitle('Strategy Returns by BTC Cycle', color='white', fontsize=13,
             fontweight='bold', y=1.03)

titles = [
    f'Cycle 3  ({cycle_price.index[0].year}-{cycle_price.index[c4_idx].year})',
    f'Cycle 4  ({cycle_price.index[c4_idx].year}-present)',
    'Full Window (Cycle 3 + 4)',
]
for ax, vals, title in zip(axes, [ret_c3_vals, ret_c4_vals, ret_full_vals], titles):
    ax.set_facecolor('#161b22')
    bars = ax.bar(strat_names, vals, color=BAR_COLORS,
                  edgecolor='#30363d', linewidth=0.6, width=0.55)
    ypad = max(abs(v) for v in vals) * 0.04
    for bar, val in zip(bars, vals):
        ytext = bar.get_height() + ypad if val >= 0 else bar.get_height() - ypad * 2
        ax.text(bar.get_x() + bar.get_width() / 2, ytext,
                f'{val:+.0f}%', ha='center', va='bottom',
                color='white', fontsize=12, fontweight='bold')
    ax.axhline(0, color='white', lw=0.6, alpha=0.35)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.tick_params(colors='white', labelsize=9)
    plt.setp(ax.get_xticklabels(), color='white')
    ax.grid(axis='y', alpha=0.12)
    for s in ax.spines.values(): s.set_color('#30363d')

plt.tight_layout()
plt.savefig('sma5_100_per_cycle_returns.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()\
"""))

# ── Assemble and write ─────────────────────────────────────────────────────────
nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'cells': cells
}

out = r'D:\Will\Learning\AI\Notebooks\SMA_DCA_BnHTest.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'OK — {len(cells)} cells written to {out}')
