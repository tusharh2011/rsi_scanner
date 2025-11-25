import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, time, date
import time as time_lib
import requests
from urllib.parse import quote
import concurrent.futures

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="Nifty Sniper Live Terminal")

# Custom CSS for "Clean & Sleek" look
st.markdown("""
<style>
    /* Compact metrics */
    .stMetric {
        background-color: #262730;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #464b5c;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stMetric label {
        font-size: 0.75rem !important;
        color: #888 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    
    /* Sleek active position card */
    .active-card {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.08) 0%, rgba(76, 175, 80, 0.04) 100%);
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 3px solid #4CAF50;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .active-card h3 {
        margin: 0;
        font-size: 1.3rem;
        line-height: 1.2;
    }
    .active-card p {
        margin: 4px 0;
        font-size: 0.85rem;
    }
    
    /* No trade card */
    .no-trade-card {
        background-color: #262730;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #464b5c;
        margin-bottom: 15px;
        text-align: center;
        color: #888;
    }
    .no-trade-card h4 {
        margin: 0 0 8px 0;
        font-size: 1rem;
    }
    .no-trade-card p {
        margin: 0;
        font-size: 0.85rem;
    }
    
    /* Compact layout */
    .block-container {
        padding-top: 1.5rem;
    }
    
    /* Info boxes */
    .stAlert {
        padding: 6px 10px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'live_trades' not in st.session_state:
    st.session_state.live_trades = []
if 'last_trade_count' not in st.session_state:
    st.session_state.last_trade_count = 0
if 'http_session' not in st.session_state:
    st.session_state.http_session = requests.Session()

# ==========================================
# 2. CONSTANTS & CONFIG
# ==========================================
DEFAULT_INSTRUMENTS = {
    "Nifty 50": "NSE_INDEX|Nifty 50",
    "Bank Nifty": "NSE_INDEX|Nifty Bank",
    "Fin Nifty": "NSE_INDEX|Nifty Fin Service",
    "Midcap Select": "NSE_INDEX|NIFTY MID SELECT", 
    "Sensex": "BSE_INDEX|SENSEX",
    "Bankex": "BSE_INDEX|BANKEX"
}

LOT_SIZES = {
    "Nifty 50": 75,
    "Bank Nifty": 35,
    "Fin Nifty": 65,
    "Midcap Select": 140,
    "Sensex": 20,
    "Bankex": 30
}

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("1. Live Connection")
    access_token = st.text_input("Upstox Access Token", type="password")
    selected_instruments = st.multiselect("Instruments", options=list(DEFAULT_INSTRUMENTS.keys()), default=["Nifty 50"])

    st.header("2. Strategy Parameters")
    rsi_len = st.number_input("RSI Length", value=14)
    anchor_lookback = st.number_input("Anchor Lookback", value=60)
    min_gap = st.number_input("Min Gap", value=10)
    deep_hook_buy = st.number_input("Deep Hook Buy (<)", value=25)
    deep_hook_sell = st.number_input("Deep Hook Sell (>)", value=75)

    st.header("3. Filters")
    check_color = st.checkbox("Strict Candle Color", value=False)
    check_mom = st.checkbox("Strict Momentum", value=False)
    check_rej = st.checkbox("Strict Wick Rejection", value=False)
    check_valley = st.checkbox("Require RSI Valley (Cross 50)", value=False)
    trigger_rsi_limit = st.number_input("Trigger RSI Limit (Buffer)", value=0.0)

    st.header("4. Execution")
    enable_sar = st.checkbox("Enable Stop & Reverse (SAR)", value=True)
    target_profit_rr = st.number_input("Target Profit (R:R)", value=30.0, step=1.0)
    sl_buffer_pts = st.number_input("Hard SL Buffer", value=20.0, step=1.0)
    be_trigger_rr = st.number_input("Breakeven Trigger", value=1.0, step=0.1)
    tier_3_action = st.selectbox("Tier 3 Action", ["Skip Trade", "Take Trade"], index=1)
    
    st.divider()
    st.caption("v2.3 - Multi-Instrument Parallel")

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def make_request(method, url, params=None, token=None, session=None):
    """Robust HTTP Request Wrapper"""
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {token}', 'Api-Version': '2.0'}
    if session is None: session = requests
    try:
        if method == "GET": response = session.get(url, headers=headers, params=params, timeout=8)
        else: response = session.post(url, headers=headers, json=params, timeout=8)
        if response.status_code == 200: return response.json()
        elif response.status_code == 429: time_lib.sleep(2.0); return None
        return None
    except Exception: return None

def fetch_live_data(token, key, session=None):
    """Fetches TODAY's intraday 1-minute candles using V3 Intraday API"""
    if not token: return pd.DataFrame()
    encoded_key = quote(key)
    # V3 Intraday Candle Data endpoint
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{encoded_key}/minutes/1"
    data = make_request("GET", url, token=token, session=session)
    if data and data.get('status') == 'success' and data.get('data', {}).get('candles'):
        candles = data['data']['candles']
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'v', 'oi'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # V3 Intraday API returns IST timestamps
        df = df.sort_values('timestamp').reset_index(drop=True)
        cols = ['open', 'high', 'low', 'close']
        df[cols] = df[cols].apply(pd.to_numeric)
        df['RSI'] = ta.rsi(df['close'], length=rsi_len)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        return df
    return pd.DataFrame()

def determine_day_type(df_day):
    if df_day.empty: return "Unknown"
    high, low = df_day['high'].max(), df_day['low'].min()
    open_p, close_p = df_day['open'].iloc[0], df_day['close'].iloc[-1]
    day_range = high - low
    if day_range > 250: return "High Volatility"
    elif abs(close_p - open_p) < (day_range * 0.25): return "Choppy / Doji"
    elif close_p > open_p: return "Bullish Trend"
    else: return "Bearish Trend"

def find_signal(i, df, row):
    if i < anchor_lookback: return None
    if (row['high'] - row['low']) > 50: return None 
    curr_t = row['timestamp'].time()
    if curr_t < time(9, 30) or curr_t > time(14, 50): return None
    prev_candle = df.iloc[i-1]

    # BUY Logic
    if df.iloc[i-1]['low'] > row['low'] and df.iloc[i+1]['low'] > row['low']:
        if row['RSI'] > (50 + trigger_rsi_limit): return None 
        if check_color and row['close'] < row['open']: return None
        if check_mom and row['close'] <= prev_candle['high']: return None
        if check_rej and (row['high'] - row['low']) > 0 and (row['close'] - row['low']) / (row['high'] - row['low']) < 0.5: return None

        for j in range(i-5, i-anchor_lookback, -1):
            prev = df.iloc[j]
            if (row['low'] < prev['low']) and (row['RSI'] > prev['RSI']):
                if (i - j) < min_gap: continue
                valley_found = any(df.iloc[k]['RSI'] >= 50 for k in range(j+1, i))
                if check_valley and not valley_found: continue 
                
                valley_idx = next((k for k in range(j+1, i) if df.iloc[k]['RSI'] >= 50), None)
                if valley_idx is not None:
                    valley_time = df.iloc[valley_idx]['timestamp']
                    valley_rsi = df.iloc[valley_idx]['RSI']
                    valley_str = valley_time.strftime('%H:%M')
                else:
                    valley_time = None
                    valley_rsi = 0.0
                    valley_str = "No Cross"

                if prev['RSI'] < deep_hook_buy:
                    if df['RSI'].iloc[j:i].max() < 70 and (row['RSI'] - prev['RSI']) > 10:
                         return {'Type': 'BUY', 'Tier': 'Tier 1', 'Logic': f"Anchor {prev['RSI']:.1f}", 'SL Price': row['low'] - sl_buffer_pts, 'Anchor Time': prev['timestamp'], 'Valley Time': valley_str, 'Anchor RSI': prev['RSI'], 'Valley RSI': valley_rsi}
                elif prev['RSI'] < 35 and tier_3_action == "Take Trade":
                     return {'Type': 'BUY', 'Tier': 'Tier 3', 'Logic': f"Weak Anchor {prev['RSI']:.1f}", 'SL Price': row['low'] - sl_buffer_pts, 'Anchor Time': prev['timestamp'], 'Valley Time': valley_str, 'Anchor RSI': prev['RSI'], 'Valley RSI': valley_rsi}

    # SELL Logic
    if df.iloc[i-1]['high'] < row['high'] and df.iloc[i+1]['high'] < row['high']:
        if row['RSI'] < (50 - trigger_rsi_limit): return None
        if check_color and row['close'] > row['open']: return None
        if check_mom and row['close'] >= prev_candle['low']: return None
        if check_rej and (row['high'] - row['low']) > 0 and (row['high'] - row['close']) / (row['high'] - row['low']) < 0.5: return None

        for j in range(i-5, i-anchor_lookback, -1):
            prev = df.iloc[j]
            if (row['high'] > prev['high']) and (row['RSI'] < prev['RSI']):
                if (i - j) < min_gap: continue
                valley_found = any(df.iloc[k]['RSI'] <= 50 for k in range(j+1, i))
                if check_valley and not valley_found: continue
                
                valley_idx = next((k for k in range(j+1, i) if df.iloc[k]['RSI'] <= 50), None)
                if valley_idx is not None:
                    valley_time = df.iloc[valley_idx]['timestamp']
                    valley_rsi = df.iloc[valley_idx]['RSI']
                    valley_str = valley_time.strftime('%H:%M')
                else:
                    valley_time = None
                    valley_rsi = 0.0
                    valley_str = "No Cross"

                if prev['RSI'] > deep_hook_sell:
                    if df['RSI'].iloc[j:i].min() > 30 and (prev['RSI'] - row['RSI']) > 10:
                         return {'Type': 'SELL', 'Tier': 'Tier 1', 'Logic': f"Anchor {prev['RSI']:.1f}", 'SL Price': row['high'] + sl_buffer_pts, 'Anchor Time': prev['timestamp'], 'Valley Time': valley_str, 'Anchor RSI': prev['RSI'], 'Valley RSI': valley_rsi}
                elif prev['RSI'] > 65 and tier_3_action == "Take Trade":
                     return {'Type': 'SELL', 'Tier': 'Tier 3', 'Logic': f"Weak Anchor {prev['RSI']:.1f}", 'SL Price': row['high'] + sl_buffer_pts, 'Anchor Time': prev['timestamp'], 'Valley Time': valley_str, 'Anchor RSI': prev['RSI'], 'Valley RSI': valley_rsi}
    return None

def analyze_instrument(name, key, token, session):
    """Analyzes a single instrument: fetches data, simulates trades, returns results."""
    df = fetch_live_data(token, key, session=session)
    if df.empty: return None
    
    # --- LOGIC SIMULATION ---
    today_date = datetime.now().date()
    today_indices = df[df['timestamp'].dt.date == today_date].index
    
    trades_today = []
    current_trade = None 
    last_closed = None
    df_today = pd.DataFrame()
    day_type = "Waiting for Data"
    
    if len(today_indices) > 0:
        df_today = df.loc[today_indices].copy()
        day_type = determine_day_type(df_today)
        today_start_idx = today_indices[0]
        start_loop = max(anchor_lookback, today_start_idx)
        
        for i in range(start_loop, len(df) - 1): 
            row = df.iloc[i]
            curr_time = row['timestamp']
            new_signal = find_signal(i, df, row)
            
            # SAR Logic
            if current_trade and enable_sar and new_signal:
                if new_signal['Type'] != current_trade['Type'] and new_signal['Tier'] == 'Tier 1':
                    current_trade['Exit Time'] = curr_time
                    current_trade['Exit Price'] = row['close']
                    current_trade['Exit Reason'] = f"SAR ({new_signal['Type']})"
                    trades_today.append(current_trade)
                    last_closed = current_trade
                    current_trade = None
            
            # Trade Management
            if current_trade:
                trade = current_trade
                if trade['Type'] == 'BUY': max_prof = row['high'] - trade['Entry Price']
                else: max_prof = trade['Entry Price'] - row['low']
                
                tgt_mult = 0.5 if trade['Tier'] == 'Tier 3' else 1.0
                tgt_pts = trade['Risk'] * target_profit_rr * tgt_mult
                
                exit = False; reason = ""; pr = 0.0
                if max_prof >= tgt_pts:
                    exit = True; reason = "Target"; pr = trade['Entry Price'] + tgt_pts if trade['Type']=='BUY' else trade['Entry Price'] - tgt_pts
                elif (trade['Type']=='BUY' and row['low'] <= trade['SL']) or (trade['Type']=='SELL' and row['high'] >= trade['SL']):
                    exit = True; reason = "Stop Loss"; pr = trade['SL']
                elif curr_time.time() >= time(15, 25):
                    exit = True; reason = "EOD"; pr = row['close']
                    
                if exit:
                    trade['Exit Time'] = curr_time
                    trade['Exit Price'] = pr
                    trade['Exit Reason'] = reason
                    trades_today.append(trade)
                    last_closed = trade
                    current_trade = None
                    continue
                
                # TSL
                if not trade['Is_BE'] and max_prof >= (trade['Risk'] * be_trigger_rr):
                    trade['SL'] = trade['Entry Price']; trade['Is_BE'] = True
                elif trade['Is_BE']:
                    ema = row['EMA_50']
                    if trade['Type']=='BUY' and row['low'] > ema:
                        if ema > trade['SL']: trade['SL'] = ema - 2
                    elif trade['Type']=='SELL' and row['high'] < ema:
                        if ema < trade['SL']: trade['SL'] = ema + 2
                        
            # Open New Trade
            if current_trade is None and new_signal:
                take = True
                if last_closed:
                    if new_signal['Type'] == last_closed['Type']:
                        pnl = (last_closed['Exit Price'] - last_closed['Entry Price']) if last_closed['Type']=='BUY' else (last_closed['Entry Price'] - last_closed['Exit Price'])
                        if pnl > 0:
                            if new_signal['Type']=='BUY' and row['close'] >= last_closed['Entry Price']: take = False
                            if new_signal['Type']=='SELL' and row['close'] <= last_closed['Entry Price']: take = False
                            if new_signal['Tier'] != 'Tier 1': take = False
                if take:
                    risk = abs(row['close'] - new_signal['SL Price'])
                    current_trade = {
                        'Instrument': name,
                        'Entry Time': curr_time, 'Type': new_signal['Type'],
                        'Tier': new_signal['Tier'], 'Entry Price': row['close'],
                        'SL': new_signal['SL Price'], 'Risk': risk,
                        'Exit Time': None, 'Exit Price': None, 'Exit Reason': "Active",
                        'Is_BE': False, 'Logic': new_signal['Logic'],
                        'Anchor Time': new_signal['Anchor Time'], 'Valley Time': new_signal['Valley Time'],
                        'Entry RSI': row['RSI'],
                        'Anchor RSI': new_signal['Anchor RSI'],
                        'Valley RSI': new_signal['Valley RSI']
                    }
                    
    # Add Instrument name to all trades
    for t in trades_today:
        t['Instrument'] = name
        
    return {
        'name': name,
        'df': df,
        'trades': trades_today,
        'current_trade': current_trade,
        'day_type': day_type,
        'df_today': df_today
    }

def render_trade_log(all_trades):
    if all_trades:
        df_res = pd.DataFrame(all_trades)
        # PnL Calculation with Dynamic Lot Size
        df_res['Lot Size'] = df_res['Instrument'].map(LOT_SIZES)
        df_res['PnL'] = df_res.apply(lambda r: ((r['Exit Price'] - r['Entry Price']) if r['Type']=='BUY' else (r['Entry Price'] - r['Exit Price'])) * r['Lot Size'] if r['Exit Price'] else 0, axis=1)
        
        # Formatting
        df_res['Entry Time'] = df_res['Entry Time'].dt.strftime('%H:%M')
        df_res['Exit Time'] = df_res['Exit Time'].apply(lambda x: x.strftime('%H:%M') if pd.notnull(x) else "Active")
        
        # Ensure new columns exist
        for col in ['Entry RSI', 'Anchor RSI', 'Valley RSI']:
            if col not in df_res.columns: df_res[col] = 0.0
            
        # Select Columns
        display_cols = ['Instrument', 'Entry Time', 'Type', 'Entry Price', 'Exit Time', 'Exit Price', 'PnL', 'Logic', 'Entry RSI', 'Valley RSI', 'Anchor RSI', 'Exit Reason']
        
        # --- Custom Table UI ---
        st.markdown("---")
        t1, t2, t3 = st.columns([3, 1, 1])
        with t1:
            st.markdown("### 📜 Full Report")
        with t2:
            sort_col = st.selectbox("Sort By", display_cols, index=1, label_visibility="collapsed")
        with t3:
            sort_asc = st.toggle("Ascending", value=False)
        
        # Sorting
        if sort_col in df_res.columns:
            df_res = df_res.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

        # Styling
        def highlight_pnl(val):
            if val > 0: return 'color: #4CAF50; font-weight: bold'
            elif val < 0: return 'color: #FF5252; font-weight: bold'
            else: return 'color: #888888'

        st.dataframe(
            df_res[display_cols].style
            .applymap(highlight_pnl, subset=['PnL'])
            .format({
                'Entry Price': '{:.2f}', 'Exit Price': '{:.2f}', 'PnL': '{:.2f}',
                'Entry RSI': '{:.1f}', 'Anchor RSI': '{:.1f}', 'Valley RSI': '{:.1f}'
            }, na_rep="-"),
            use_container_width=True,
            height=400
        )
        
        # Notifications
        if len(all_trades) > st.session_state.last_trade_count:
            # Find the new trades
            new_count = len(all_trades) - st.session_state.last_trade_count
            # Just notify about the last one for simplicity or iterate
            last_trade_row = df_res.iloc[-1]
            if last_trade_row['Exit Time'] != "Active":
                st.toast(f"{last_trade_row['Instrument']} Trade Closed: {last_trade_row['Type']} PnL: {last_trade_row['PnL']:.2f}", icon="💰")
            st.session_state.last_trade_count = len(all_trades)
    else:
        st.markdown("### 📜 Full Report")
        st.info("No trades executed today yet.")

# ==========================================
# 5. LIVE MONITOR UI LAYOUT
# ==========================================
# Header
c_head_1, c_head_2 = st.columns([3, 1])
with c_head_1:
    st.title("⚡ Nifty Sniper Multi")
with c_head_2:
    st.markdown(f"<div style='text-align: right; padding-top: 20px; color: #888;'>{datetime.now().strftime('%H:%M:%S')} IST</div>", unsafe_allow_html=True)

# Controls
c_ctrl_1, c_ctrl_2, c_ctrl_3 = st.columns([1, 1, 3])
with c_ctrl_1:
    scan_btn = st.button("🔎 Scan Market", use_container_width=True, type="primary")
with c_ctrl_2:
    auto_refresh = st.checkbox("Auto-Refresh")
with c_ctrl_3:
    if auto_refresh:
        refresh_sec = st.slider("Refresh Interval (s)", 5, 60, 15, label_visibility="collapsed")

# Layout Containers
active_section = st.container()
metrics_section = st.container()
log_section = st.container()

if scan_btn or auto_refresh:
    if not selected_instruments:
        st.error("Please select at least one instrument.")
    else:
        with st.spinner(f"Scanning {len(selected_instruments)} Instruments..."):
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create futures for all selected instruments
                future_to_name = {
                    executor.submit(analyze_instrument, name, DEFAULT_INSTRUMENTS[name], access_token, st.session_state.http_session): name 
                    for name in selected_instruments
                }
                
                for future in concurrent.futures.as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        res = future.result()
                        if res: results.append(res)
                    except Exception as e:
                        st.error(f"Error analyzing {name}: {e}")

            # Aggregate Results
            all_active_trades = []
            all_closed_trades = []
            
            for res in results:
                if res['current_trade']:
                    all_active_trades.append((res['name'], res['current_trade'], res['df']))
                all_closed_trades.extend(res['trades'])

            # --- RENDER UI ---
            
            # 1. Active Position Cards
            with active_section:
                if all_active_trades:
                    cols = st.columns(2)
                    for idx, (name, trade, df) in enumerate(all_active_trades):
                        with cols[idx % 2]:
                            curr_price = df.iloc[-1]['close']
                            price_diff = (curr_price - trade['Entry Price']) if trade['Type']=='BUY' else (trade['Entry Price'] - curr_price)
                            lot_size = LOT_SIZES.get(name, 1)
                            pnl_value = price_diff * lot_size
                            pnl_color = "#4CAF50" if pnl_value > 0 else "#FF5252"
                            
                            st.markdown(f'''
                            <div class="active-card">
                                <div style="margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px;">
                                    <span style="color: #fff; font-weight: bold; font-size: 1.1rem;">{name}</span>
                                </div>
                                <div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;">
                                    <div style="flex: 0 0 70px;">
                                        <p style="color: #888; margin: 0; font-size: 0.7rem;">Time</p>
                                        <h3 style="color: #4CAF50; margin: 2px 0;">{trade['Entry Time'].strftime('%H:%M')}</h3>
                                        <p style="color: #666; margin: 0; font-size: 0.7rem;">{trade['Tier']}</p>
                                    </div>
                                    <div style="flex: 0 0 60px; text-align: center;">
                                        <p style="color: #888; margin: 0; font-size: 0.7rem;">Type</p>
                                        <h3 style="color: #4CAF50; margin: 2px 0; font-size: 1.2rem;">{trade['Type']}</h3>
                                    </div>
                                    <div style="flex: 0 0 85px;">
                                        <p style="color: #888; margin: 0; font-size: 0.7rem;">Entry</p>
                                        <p style="font-size: 1.05rem; font-weight: 600; margin: 2px 0;">{trade['Entry Price']:.2f}</p>
                                    </div>
                                    <div style="flex: 0 0 85px;">
                                        <p style="color: #888; margin: 0; font-size: 0.7rem;">Spot</p>
                                        <p style="font-size: 1.05rem; font-weight: 600; margin: 2px 0;">{curr_price:.2f}</p>
                                        <p style="color: {pnl_color}; margin: 0; font-size: 0.7rem;">{price_diff:+.2f}</p>
                                    </div>
                                    <div style="flex: 0 0 95px;">
                                        <p style="color: #888; margin: 0; font-size: 0.7rem;">PnL (₹)</p>
                                        <p style="font-size: 1.25rem; font-weight: bold; color: {pnl_color}; margin: 2px 0;">{pnl_value:+.2f}</p>
                                    </div>
                                    <div style="flex: 0 0 110px; padding-left: 10px; border-left: 1px solid #464b5c;">
                                        <p style="color: #888; margin: 0 0 2px 0; font-size: 0.7rem;">Entry: {trade.get('Entry RSI', 0):.1f}</p>
                                        <p style="color: #888; margin: 0 0 2px 0; font-size: 0.7rem;">Anchor: {trade.get('Anchor RSI', 0):.1f}</p>
                                        <p style="color: #888; margin: 0; font-size: 0.7rem;">Valley: {trade.get('Valley RSI', 0):.1f}</p>
                                    </div>
                                    <div style="flex: 1; padding-left: 10px; border-left: 1px solid #464b5c;">
                                        <p style="color: #888; margin: 0; font-size: 0.7rem;">Logic</p>
                                        <p style="font-weight: 600; margin: 2px 0; font-size: 0.85rem;">{trade['Logic']}</p>
                                        <p style="color: #FF5252; margin: 0; font-size: 0.7rem;">SL: {trade['SL']:.2f}</p>
                                    </div>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-trade-card"><h4>⚪ No Active Positions</h4><p>Scanning for setups...</p></div>', unsafe_allow_html=True)

            # 2. Market Metrics (Grid)
            with metrics_section:
                if results:
                    cols = st.columns(len(results))
                    for idx, res in enumerate(results):
                        with cols[idx]:
                            if not res['df_today'].empty:
                                curr_price = res['df_today'].iloc[-1]['close']
                                day_open = res['df_today'].iloc[0]['open']
                                day_high = res['df_today']['high'].max()
                                day_low = res['df_today']['low'].min()
                                curr_rsi = res['df_today'].iloc[-1]['RSI']
                                
                                change = curr_price - day_open
                                pct_change = (change / day_open) * 100
                                
                                st.markdown(f"**{res['name']}**")
                                st.caption(res['day_type'])
                                st.metric("Spot", f"{curr_price:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
                                
                                st.markdown(f"""
                                <div style="font-size: 0.75rem; color: #aaa; margin-bottom: 5px;">
                                    <div>Open: <span style="color: #ddd;">{day_open:.2f}</span></div>
                                    <div>H: <span style="color: #ddd;">{day_high:.0f}</span> L: <span style="color: #ddd;">{day_low:.0f}</span></div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.metric("RSI", f"{curr_rsi:.1f}")
                            else:
                                st.warning(f"{res['name']}: No Data")
                else:
                    st.warning("Waiting for market data...")

            # 3. Trade Log
            with log_section:
                render_trade_log(all_closed_trades)

    # Auto Refresh Loop
    if auto_refresh:
        time_lib.sleep(refresh_sec)
        st.rerun()
