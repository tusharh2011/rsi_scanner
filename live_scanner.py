import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, time as dt_time, timedelta, date
from urllib.parse import quote
import time
import concurrent.futures
import os
import csv

# ==========================================
# PAGE CONFIG & SESSION STATE
# ==========================================
st.set_page_config(page_title="Multi-Index Live Scanner", layout="wide", page_icon="📡")

if 'active_trades' not in st.session_state:
    st.session_state.active_trades = {}
if 'market_snapshot' not in st.session_state:
    st.session_state.market_snapshot = {}
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
LOG_FILE = "trade_history.csv"

LOT_SIZES = {
    "Nifty 50": 25,
    "Bank Nifty": 15,
    "Fin Nifty": 25,
    "Midcap Select": 50, 
    "Sensex": 10,
    "Bankex": 15
}

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("📡 Option Scalper Config")

# 1. Connection
with st.sidebar.expander("🔐 Connection", expanded=True):
    api_token = st.text_input("Upstox Access Token", type="password")

# 2. Instrument Selection
with st.sidebar.expander("📊 Instruments", expanded=True):
    DEFAULT_INSTRUMENTS = {
        "Nifty 50": "NSE_INDEX|Nifty 50",
        "Bank Nifty": "NSE_INDEX|Nifty Bank",
        "Fin Nifty": "NSE_INDEX|Nifty Fin Service",
        "Midcap Select": "NSE_INDEX|Nifty Midcap Select",
        "Sensex": "BSE_INDEX|SENSEX",
        "Bankex": "BSE_INDEX|BANKEX"
    }
    
    selected_names = st.multiselect(
        "Select Indices", 
        list(DEFAULT_INSTRUMENTS.keys()), 
        default=["Nifty 50", "Bank Nifty"]
    )
    scan_list = {name: DEFAULT_INSTRUMENTS[name] for name in selected_names}

# 3. Option Selection
with st.sidebar.expander("🎯 Option Selection", expanded=True):
    itm_depth = st.selectbox("ITM Depth", [0, 100, 200], index=1)
    target_delta = st.slider("Preferred Delta", 0.4, 0.9, 0.65)

# 4. Strategy Config
with st.sidebar.expander("🧠 Strategy Logic", expanded=False):
    enable_rsi = st.checkbox("Enable RSI Strategy", value=True)
    rsi_period = st.number_input("RSI Period", value=14)
    div_limit = st.number_input("Max RSI for Entry", value=50)
    enable_cpr = st.checkbox("Enable CPR Strategy", value=True)
    use_vwap = st.checkbox("VWAP Confirmation", value=True)

# 5. Risk Management
with st.sidebar.expander("💰 Risk & Targets", expanded=False):
    target_pts = st.number_input("Target (Spot Pts)", value=20.0)
    stop_pts = st.number_input("Stop Loss (Spot Pts)", value=25.0)
    use_trailing = st.checkbox("Enable Trailing", value=True)
    trail_start = st.number_input("Trail Start (Profit)", value=10.0)
    trail_offset = st.number_input("Trail Offset", value=10.0)

refresh_rate = st.sidebar.slider("Refresh Rate (sec)", 10, 120, 60)

# ==========================================
# CORE FUNCTIONS
# ==========================================

def log_message(msg, type="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.insert(0, {"time": timestamp, "msg": msg, "type": type})

def save_trade_to_csv(trade_data):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Date", "Instrument", "Strategy", "Entry Time", "Exit Time", "Status", "Spot Entry", "Spot Exit", "Spot PnL", "Option", "Opt Entry", "Opt Exit", "Opt PnL", "Profit (INR)"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d"), trade_data['instrument'], trade_data['reason'], trade_data['entry_time'], datetime.now().strftime("%H:%M:%S"),
            trade_data['exit_status'], f"{trade_data['entry_spot']:.2f}", f"{trade_data['exit_spot']:.2f}", f"{trade_data['spot_pnl']:.2f}",
            f"{trade_data['opt_strike']} CE", f"{trade_data['opt_entry']:.2f}", f"{trade_data['opt_exit']:.2f}", f"{trade_data['opt_pnl']:.2f}", f"{trade_data['pnl_inr']:.2f}"
        ])

def fetch_recent_data(token, key, days=5):
    # FIX 1: Set end_date to TOMORROW to capture TODAY'S live candles
    # Upstox 'to_date' is exclusive or requires future date for current day data
    end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # Encode key for URL (handles spaces in 'Nifty 50')
    encoded_key = quote(key)
    
    url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/1minute/{end_date}/{start_date}"
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {token}'}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        if data.get('status') == 'success' and data.get('data'):
            candles = data['data']['candles']
            if not candles: return pd.DataFrame()
            df = pd.DataFrame(candles, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
            df.sort_index(ascending=True, inplace=True)
            return df
    except: pass
    return pd.DataFrame()

def get_live_quote(token, instrument_key):
    # FIX 2: Use 'params' dict to handle URL encoding of keys like 'NSE_INDEX|Nifty 50'
    url = "https://api.upstox.com/v2/market-quote/ltp"
    params = {'instrument_key': instrument_key}
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {token}'}
    
    try:
        res = requests.get(url, headers=headers, params=params, timeout=3)
        data = res.json()
        if data['status'] == 'success':
            # API returns: data: { "NSE_INDEX|Nifty 50": { "last_price": 24000.00 } }
            return data['data'].get(instrument_key, {}).get('last_price', 0)
    except: pass
    return 0

def calculate_indicators(df):
    if df.empty: return df
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + (gain / loss.fillna(0))))
    df['SMA_200'] = df['Close'].rolling(200).mean()
    
    df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical'] * df['Volume']
    df['Cum_VP'] = df.groupby(df.index.date)['VP'].cumsum()
    df['Cum_Vol'] = df.groupby(df.index.date)['Volume'].cumsum()
    df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']
    return df

def prepare_cpr(df):
    if df.empty: return df
    daily = df.resample('D').agg({'High':'max','Low':'min','Close':'last'}).dropna()
    daily['P'] = (daily['High'] + daily['Low'] + daily['Close']) / 3
    daily['BC'] = (daily['High'] + daily['Low']) / 2
    daily['TC'] = 2*daily['P'] - daily['BC']
    
    daily_shifted = daily.shift(1)
    df['Date'] = df.index.date
    daily_shifted['Date'] = daily_shifted.index.date
    merged = df.merge(daily_shifted[['Date','P','BC','TC']], on='Date', how='left')
    merged.index = df.index
    return merged

def get_option_contract(token, instrument_key, spot_price, signal_type, depth):
    today = date.today()
    days_ahead = 3 - today.weekday()
    if days_ahead < 0: days_ahead += 7
    expiry_date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    url = "https://api.upstox.com/v2/option/chain"
    params = {'instrument_key': instrument_key, 'expiry_date': expiry_date}
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {token}'}
    
    try:
        res = requests.get(url, headers=headers, params=params, timeout=5)
        data = res.json()
        if data.get('status') != 'success': return None
        options = data.get('data', [])
        
        if "Sensex" in instrument_key or "Bank" in instrument_key: step = 100
        elif "Midcap" in instrument_key: step = 25 
        else: step = 50
            
        rounded_spot = round(spot_price / step) * step
        target_strike = rounded_spot - depth if signal_type == "BUY" else rounded_spot + depth
        
        for opt in options:
            if opt.get('strike_price') == target_strike:
                if signal_type == "BUY" and 'call_options' in opt:
                    d = opt['call_options']
                    return {'key': d['instrument_key'], 'ltp': d['market_data']['ltp'], 'delta': d['option_greeks']['delta'], 'strike': target_strike}
    except: return None
    return None

def process_instrument(name, key, token, strategy_config, option_config):
    result = { "name": name, "key": key, "price": 0.0, "status": "Wait", "signal": False, "reason": "", "option": None }
    
    # Get Live Quote First (More reliable for current price than candle)
    live_ltp = get_live_quote(token, key)
    
    df = fetch_recent_data(token, key)
    if df.empty: 
        # Fallback if history fails but LTP works
        if live_ltp > 0: result["price"] = live_ltp
        return result
        
    df = calculate_indicators(df)
    df = prepare_cpr(df)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Use live LTP if available, else candle close
    current_spot = live_ltp if live_ltp > 0 else last['Close']
    result["price"] = current_spot
    
    signal = False
    reason = ""
    
    if strategy_config['enable_rsi']:
        idx = -1 - strategy_config['pivot_window']
        if len(df) > abs(idx) + 5:
            window = df['Low'].iloc[idx - strategy_config['pivot_window'] : idx + strategy_config['pivot_window'] + 1]
            if df['Low'].iloc[idx] == window.min() and df['RSI'].iloc[idx] < strategy_config['div_limit']:
                signal = True; reason = "RSI Pivot"

    if strategy_config['enable_cpr'] and not signal:
        if last['Close'] > last['TC']:
            if not strategy_config['use_vwap'] or (last['Close'] > last['VWAP']):
                if prev['Close'] <= last['TC']:
                    signal = True; reason = "CPR Breakout"

    result["signal"] = signal
    result["reason"] = reason
    if signal:
        opt_data = get_option_contract(token, key, current_spot, "BUY", option_config['depth'])
        if opt_data: result['option'] = opt_data
            
    return result

# ==========================================
# MAIN LOGIC
# ==========================================

def run_parallel_scan():
    if not api_token: st.error("Token missing!"); return

    strat_cfg = {'enable_rsi': enable_rsi, 'rsi_period': rsi_period, 'pivot_window': 5, 'div_limit': div_limit, 'enable_cpr': enable_cpr, 'use_vwap': use_vwap}
    opt_cfg = {'depth': itm_depth}

    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for name, key in scan_list.items():
            futures.append(executor.submit(process_instrument, name, key, api_token, strat_cfg, opt_cfg))
    
    for future in concurrent.futures.as_completed(futures):
        res = future.result()
        name = res['name']
        
        st.session_state.market_snapshot[name] = { 'price': res['price'], 'status': res['reason'] if res['signal'] else "Scanning", 'time': datetime.now().strftime("%H:%M:%S") }
        
        if res['signal'] and name not in st.session_state.active_trades:
            spot = res['price']
            opt = res['option']
            st.session_state.active_trades[name] = {
                'instrument': name,
                'entry_spot': spot, 'target_spot': spot + target_pts, 'stop_spot': spot - stop_pts,
                'highest_spot': spot, 'opt_key': opt['key'] if opt else "N/A",
                'opt_entry': opt['ltp'] if opt else 0, 'opt_strike': opt['strike'] if opt else 0,
                'opt_delta': opt['delta'] if opt else 0.5,
                'lot_size': LOT_SIZES.get(name, 25), 'reason': res['reason'],
                'entry_time': datetime.now().strftime("%H:%M:%S")
            }
            log_message(f"🚀 ENTRY {name} | Spot: {spot} | {res['reason']}", "success")
            st.toast(f"{name}: Buy Signal!", icon="🚀")

    # MANAGEMENT
    active_keys = list(st.session_state.active_trades.keys())
    for name in active_keys:
        trade = st.session_state.active_trades[name]
        
        # Fetch fresh prices for management
        curr_spot = get_live_quote(api_token, scan_list[name]) # Force fetch fresh spot
        if curr_spot == 0: curr_spot = trade['entry_spot'] # Fallback
        
        # Option Price
        curr_opt = trade['opt_entry']
        if trade['opt_key'] != "N/A":
            quote = get_live_quote(api_token, trade['opt_key'])
            if quote > 0: curr_opt = quote
            
        spot_pnl = curr_spot - trade['entry_spot']
        opt_pnl = curr_opt - trade['opt_entry']
        pnl_inr = opt_pnl * trade['lot_size']
        
        # Trailing
        if use_trailing:
            if curr_spot > trade['highest_spot']:
                trade['highest_spot'] = curr_spot
                if (trade['highest_spot'] - trade['entry_spot']) >= trail_start:
                    new_sl = trade['highest_spot'] - trail_offset
                    if new_sl > trade['stop_spot']:
                        trade['stop_spot'] = new_sl
                        log_message(f"{name}: Trailing SL -> {new_sl:.2f}")

        # Exits
        exit_msg = ""
        is_exit = False
        if curr_spot >= trade['target_spot']:
            exit_msg = "Target Hit"; is_exit = True
        elif curr_spot <= trade['stop_spot']:
            exit_msg = "SL Hit"; is_exit = True
            
        if is_exit:
            log_message(f"{name}: {exit_msg} | PnL: ₹{pnl_inr:.0f}", "success" if pnl_inr>0 else "error")
            st.toast(f"{name}: {exit_msg}", icon="💰")
            
            trade['exit_spot'] = curr_spot
            trade['spot_pnl'] = spot_pnl
            trade['opt_exit'] = curr_opt
            trade['opt_pnl'] = opt_pnl
            trade['pnl_inr'] = pnl_inr
            trade['exit_status'] = exit_msg
            
            save_trade_to_csv(trade)
            del st.session_state.active_trades[name]
        else:
            trade['curr_opt_price'] = curr_opt
            trade['opt_pnl'] = opt_pnl
            trade['pnl_inr'] = pnl_inr

    st.session_state.last_scan_time = datetime.now().strftime("%H:%M:%S")

# ==========================================
# UI
# ==========================================
st.title("📡 Option Scalper (Delta + IV)")

c1, c2 = st.columns([3, 1])
with c1: st.caption(f"Scanning: {', '.join(scan_list.keys())} | Last: {st.session_state.last_scan_time}")
with c2: 
    if st.button("🔄 Scan Now", use_container_width=True): run_parallel_scan()

st.divider()

st.subheader("🟢 Active Positions")
if st.session_state.active_trades:
    for name, t in st.session_state.active_trades.items():
        with st.container():
            col1, col2, col3, col4, col5 = st.columns(5)
            # Use fresh fetched spot if available, else last snapshot
            curr = get_live_quote(api_token, scan_list[name])
            if curr == 0: curr = t['entry_spot']
            
            col1.metric(name, f"{curr:.2f}")
            col2.metric("Option", f"{t['opt_strike']} CE", f"Entry: {t['opt_entry']}")
            col3.metric("Premium", f"{t.get('curr_opt_price',0):.2f}", delta=f"{t.get('opt_pnl',0):.2f}")
            col4.metric("P&L (₹)", f"₹ {t.get('pnl_inr',0):.0f}")
            col5.metric("Delta", f"{t.get('opt_delta',0):.2f}")
            st.progress(min(max((curr - t['entry_spot']) / (t['target_spot'] - t['entry_spot']), 0.0), 1.0))
            st.markdown("---")
else:
    st.info("Waiting for signals...")

st.divider()

st.subheader("📊 Market Watch")
if st.session_state.market_snapshot:
    cols = st.columns(4)
    for i, (name, data) in enumerate(st.session_state.market_snapshot.items()):
        cols[i%4].metric(name, f"{data['price']:.2f}", data['status'])

with st.expander("📜 Logs & History", expanded=True):
    for l in st.session_state.logs[:5]:
        st.write(f"**{l['time']}** : {l['msg']}")
    
    if os.path.isfile(LOG_FILE):
        st.markdown("---")
        st.markdown("### 📁 Trade History")
        st.dataframe(pd.read_csv(LOG_FILE).tail(10))

if st.sidebar.checkbox("Auto-Refresh"):
    time.sleep(refresh_rate)
    st.rerun()

if st.session_state.last_scan_time is None and api_token:
    run_parallel_scan()
