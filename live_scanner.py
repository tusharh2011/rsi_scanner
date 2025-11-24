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
if 'todays_history' not in st.session_state:
    st.session_state.todays_history = []

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
LOG_FILE = "trade_history.csv"

# EXACT LOT SIZES
LOT_SIZES = {
    "Nifty 50": 75,
    "Bank Nifty": 30,
    "Fin Nifty": 65,
    "Midcap Select": 75,
    "Sensex": 10,
    "Bankex": 15
}

# Override with user provided specific sizes if needed
LOT_SIZES = {
    "Nifty 50": 75,
    "Bank Nifty": 35, 
    "Fin Nifty": 65,
    "Midcap Select": 140, 
    "Sensex": 20,
    "Bankex": 30
}

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("📡 Scanner Config")

st.sidebar.header("🔌 Connection")
api_token = st.sidebar.text_input("Upstox Token", type="password", placeholder="Enter Access Token")

st.sidebar.header("⏱️ Cycle Control")
auto_refresh = st.sidebar.toggle("Enable Auto-Refresh", value=False)
refresh_rate = st.sidebar.slider("Scan Interval (sec)", 10, 120, 60)

with st.sidebar.expander("📊 Instruments", expanded=True):
    DEFAULT_INSTRUMENTS = {
        "Nifty 50": "NSE_INDEX|Nifty 50",
        "Bank Nifty": "NSE_INDEX|Nifty Bank",
        "Fin Nifty": "NSE_INDEX|Nifty Fin Service",
        "Midcap Select": "NSE_INDEX|NIFTY MID SELECT", 
        "Sensex": "BSE_INDEX|SENSEX",
        "Bankex": "BSE_INDEX|BANKEX"
    }
    selected_names = st.multiselect("Select Indices", list(DEFAULT_INSTRUMENTS.keys()), default=["Nifty 50", "Bank Nifty"])
    scan_list = {name: DEFAULT_INSTRUMENTS[name] for name in selected_names}

with st.sidebar.expander("🎯 Option Logic", expanded=True):
    itm_depth = st.selectbox("ITM Depth", [0, 100, 200], index=2) 
    lot_multiplier = st.number_input("Lot Multiplier", min_value=1, value=1, step=1, help="Total Qty = Lot Size * Multiplier")

with st.sidebar.expander("🧠 Strategies", expanded=False):
    enable_rsi = st.checkbox("RSI Strategy", value=True)
    rsi_period = st.number_input("RSI Period", value=14)
    div_limit = st.number_input("Max RSI for Entry", value=50)
    enable_cpr = st.checkbox("CPR Strategy", value=True)
    use_vwap = st.checkbox("VWAP Confirmation", value=True)

with st.sidebar.expander("💰 Risk & Targets", expanded=False):
    target_pts = st.number_input("Target (Spot Pts)", value=20.0)
    stop_pts = st.number_input("Stop Loss (Spot Pts)", value=25.0)
    use_trailing = st.checkbox("Enable Trailing", value=True)
    trail_start = st.number_input("Trail Start (Profit)", value=10.0)
    trail_offset = st.number_input("Trail Offset", value=10.0)

# ==========================================
# CORE FUNCTIONS
# ==========================================

def log_message(msg, type="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    if st.session_state.logs and st.session_state.logs[0]['msg'] == msg: return
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
            f"{trade_data['opt_strike']} {trade_data.get('side', 'CE')}", f"{trade_data['opt_entry']:.2f}", f"{trade_data['opt_exit']:.2f}", f"{trade_data['opt_pnl']:.2f}", f"{trade_data['pnl_inr']:.2f}"
        ])

def fetch_recent_data(token, key):
    encoded_key = quote(key)
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_key}/1minute"
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
    url = "https://api.upstox.com/v2/market-quote/ltp"
    params = {'instrument_key': instrument_key}
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {token}'}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=3)
        data = res.json()
        if data['status'] == 'success':
            return data['data'].get(instrument_key, {}).get('last_price', 0.0)
    except: pass
    return 0.0

# --- BATCH FETCH FUNCTION ---
def get_live_quotes_batch(token, keys_list):
    if not keys_list: return {}
    # Upstox supports comma separated keys
    keys_str = ",".join(keys_list)
    url = "https://api.upstox.com/v2/market-quote/ltp"
    params = {'instrument_key': keys_str}
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {token}'}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=3)
        data = res.json()
        if data['status'] == 'success':
            # Returns map: key -> last_price
            result_map = {}
            for k, v in data['data'].items():
                result_map[k] = v.get('last_price', 0.0)
            return result_map
    except: pass
    return {}

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

def get_expiry_date(instrument_name):
    today = date.today()
    weekday = today.weekday()
    if "Midcap" in instrument_name: target = 0
    elif "Fin" in instrument_name: target = 1
    elif "Bank" in instrument_name: target = 2
    elif "Nifty" in instrument_name: target = 3
    elif "Sensex" in instrument_name or "Bankex" in instrument_name: target = 4
    else: target = 3
    days_ahead = target - weekday
    if days_ahead <= 0: 
        if days_ahead < 0: days_ahead += 7
    return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

def get_option_contract(token, instrument_key, spot_price, signal_type, depth):
    expiry_date = get_expiry_date(instrument_key)
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
        if signal_type == "BUY":
            target_strike = rounded_spot - depth
            opt_type = "call_options"
        else:
            target_strike = rounded_spot + depth
            opt_type = "put_options"
        for opt in options:
            if opt.get('strike_price') == target_strike:
                if opt_type in opt:
                    d = opt[opt_type]
                    return {'key': d['instrument_key'], 'ltp': d['market_data']['ltp'], 'delta': d['option_greeks']['delta'], 'strike': target_strike, 'type': 'CE' if signal_type == "BUY" else 'PE'}
    except: return None
    return None

def get_historical_price_at_time(df, target_time):
    if df.empty: return 0
    try:
        if target_time in df.index: return df.loc[target_time]['Close']
        else:
            idx = df.index.get_indexer([target_time], method='pad')[0]
            if idx != -1: return df.iloc[idx]['Close']
    except: pass
    return 0

def check_signal(df, i, strategy_config):
    signal = False; reason = ""; sig_type = ""
    if strategy_config['enable_rsi']:
        idx = i - strategy_config['pivot_window']
        if idx >= strategy_config['pivot_window']:
            window = df['Low'].iloc[idx - strategy_config['pivot_window'] : idx + strategy_config['pivot_window'] + 1]
            if df['Low'].iloc[idx] == window.min() and df['RSI'].iloc[idx] < strategy_config['div_limit']:
                signal = True; reason = "RSI Bullish Pivot"; sig_type = "BUY"
    if strategy_config['enable_cpr'] and not signal:
        curr = df.iloc[i]; prev = df.iloc[i-1]
        if 'TC' in curr and not pd.isna(curr['TC']):
            if curr['Close'] > curr['TC']:
                if not strategy_config['use_vwap'] or (curr['Close'] > curr['VWAP']):
                    if prev['Close'] <= curr['TC']:
                        signal = True; reason = "CPR Breakout Buy"; sig_type = "BUY"
            elif curr['Close'] < curr['BC']:
                if not strategy_config['use_vwap'] or (curr['Close'] < curr['VWAP']):
                    if prev['Close'] >= curr['BC']:
                        signal = True; reason = "CPR Breakdown Sell"; sig_type = "SELL"
    return signal, reason, sig_type

def process_instrument_history(name, key, token, strategy_config, option_config, risk_config):
    result = { "name": name, "key": key, "price": 0.0, "status": "Wait", "live_signal": False, "reason": "", "live_type": "", "option": None, "history": [], "restored_trades": [] }
    live_ltp = get_live_quote(token, key)
    df = fetch_recent_data(token, key)
    if df.empty: 
        if live_ltp > 0: result["price"] = live_ltp
        return result
    df = calculate_indicators(df)
    df = prepare_cpr(df)
    last = df.iloc[-1]
    current_spot = live_ltp if live_ltp > 0 else last['Close']
    result["price"] = current_spot
    if last.name.date() < datetime.now().date(): result["status"] = "Stale Data"; return result
    today_df = df[df.index.date == datetime.now().date()]
    
    for i in range(len(today_df) - 1):
        idx_in_main_df = df.index.get_loc(today_df.index[i])
        signal, reason, sig_type = check_signal(df, idx_in_main_df, strategy_config)
        if signal:
            entry_time = today_df.index[i]
            entry_price = today_df['Close'].iloc[i]
            target = entry_price + risk_config['target'] if sig_type == "BUY" else entry_price - risk_config['target']
            stop = entry_price - risk_config['stop'] if sig_type == "BUY" else entry_price + risk_config['stop']
            status = "Open"; exit_price = 0; exit_time_dt = None
            for j in range(i + 1, len(today_df)):
                row = today_df.iloc[j]
                if sig_type == "BUY":
                    if row['High'] >= target: status = "Target Hit"; exit_price = target; exit_time_dt = row.name; break
                    elif row['Low'] <= stop: status = "SL Hit"; exit_price = stop; exit_time_dt = row.name; break
                else:
                    if row['Low'] <= target: status = "Target Hit"; exit_price = target; exit_time_dt = row.name; break
                    elif row['High'] >= stop: status = "SL Hit"; exit_price = stop; exit_time_dt = row.name; break
            if status == "Open":
                if sig_type == "BUY":
                    if current_spot >= target: status = "Target Hit"; exit_price = target; exit_time_dt = datetime.now()
                    elif current_spot <= stop: status = "SL Hit"; exit_price = stop; exit_time_dt = datetime.now()
                else:
                    if current_spot <= target: status = "Target Hit"; exit_price = target; exit_time_dt = datetime.now()
                    elif current_spot >= stop: status = "SL Hit"; exit_price = stop; exit_time_dt = datetime.now()
            
            opt_meta = get_option_contract(token, key, entry_price, sig_type, option_config['depth'])
            opt_str = "N/A"; opt_entry = 0; opt_exit = 0; opt_pnl = 0; pnl_inr = 0
            
            if opt_meta:
                opt_str = f"{opt_meta['strike']} {opt_meta['type']}"
                opt_df = fetch_recent_data(token, opt_meta['key'])
                if not opt_df.empty:
                    opt_entry = get_historical_price_at_time(opt_df, entry_time)
                    if exit_time_dt: opt_exit = get_historical_price_at_time(opt_df, exit_time_dt)
                    elif status == "Open": opt_exit = opt_meta['ltp'] # Current Price
                    
                    if opt_entry > 0 and opt_exit > 0:
                        opt_pnl = opt_exit - opt_entry
                        
                        lot_sz = int(LOT_SIZES.get(name, 25))
                        multiplier = int(risk_config['multiplier'])
                        pnl_inr = opt_pnl * lot_sz * multiplier
            
            exit_time_str = exit_time_dt.strftime("%H:%M") if exit_time_dt else "-"
            
            result["history"].append({
                "Time": entry_time.strftime("%H:%M"), "Instrument": name, "Signal": f"{sig_type} ({reason})",
                "Status": status, "Option": opt_str, "Opt Entry": f"{opt_entry:.2f}", "Exit Time": exit_time_str,
                "Opt Exit": f"{opt_exit:.2f}", "Opt PnL": f"{opt_pnl:.2f}", "Profit (INR)": f"{pnl_inr:.0f}"
            })
            
            if status == "Open" and opt_meta:
                restored_trade = {
                    'instrument': name, 'entry_spot': entry_price, 'target_spot': target, 'stop_spot': stop, 'highest_spot': current_spot,
                    'opt_key': opt_meta['key'], 'opt_entry': opt_entry if opt_entry > 0 else opt_meta['ltp'], 
                    'opt_strike': opt_meta['strike'], 'opt_delta': opt_meta['delta'] if opt_meta['delta'] else 0.5,
                    'side': opt_meta['type'], 
                    'lot_size': int(LOT_SIZES.get(name, 25)), 
                    'reason': f"{reason} (Restored)", 'entry_time': entry_time.strftime("%H:%M:%S")
                }
                result["restored_trades"].append(restored_trade)
    
    live_signal, live_reason, live_type = check_signal(df, len(df)-1, strategy_config)
    result["live_signal"] = live_signal; result["reason"] = live_reason; result["live_type"] = live_type
    if live_signal:
        opt_data = get_option_contract(token, key, current_spot, live_type, option_config['depth'])
        if opt_data: result['option'] = opt_data
    return result

# ==========================================
# MAIN LOGIC
# ==========================================

def run_parallel_scan():
    if not api_token: st.error("Token missing!"); return
    strat_cfg = {'enable_rsi': enable_rsi, 'rsi_period': rsi_period, 'pivot_window': 5, 'div_limit': div_limit, 'enable_cpr': enable_cpr, 'use_vwap': use_vwap}
    opt_cfg = {'depth': itm_depth}
    risk_cfg = {'target': target_pts, 'stop': stop_pts, 'multiplier': lot_multiplier}
    
    futures = []
    fresh_history = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for name, key in scan_list.items():
            futures.append(executor.submit(process_instrument_history, name, key, api_token, strat_cfg, opt_cfg, risk_cfg))
    
    for future in concurrent.futures.as_completed(futures):
        res = future.result()
        name = res['name']
        st.session_state.market_snapshot[name] = { 'price': res['price'], 'status': res['reason'] if res['live_signal'] else "Scanning", 'time': datetime.now().strftime("%H:%M:%S") }
        
        for r_trade in res['restored_trades']:
            if name not in st.session_state.active_trades:
                st.session_state.active_trades[name] = r_trade
                log_message(f"🔄 RESTORED {name} {r_trade['side']} | Spot: {r_trade['entry_spot']}", "info")
        
        if res['live_signal'] and name not in st.session_state.active_trades:
            spot = res['price']; opt = res['option']
            st.session_state.active_trades[name] = {
                'instrument': name, 'entry_spot': spot, 
                'target_spot': spot + target_pts if res['live_type']=="BUY" else spot - target_pts,
                'stop_spot': spot - stop_pts if res['live_type']=="BUY" else spot + stop_pts,
                'highest_spot': spot, 'opt_key': opt['key'] if opt else "N/A", 'opt_entry': opt['ltp'] if opt else 0, 
                'opt_strike': opt['strike'] if opt else 0, 'opt_delta': opt['delta'] if opt else 0.5,
                'side': opt['type'] if opt else "N/A", 
                'lot_size': int(LOT_SIZES.get(name, 25)), 
                'reason': res['reason'], 'entry_time': datetime.now().strftime("%H:%M:%S")
            }
            log_message(f"🚀 {res['live_type']} {name} | Spot: {spot} | {res['reason']}", "success")
            st.toast(f"🚀 {name}: Signal!", icon="🚀")
        
        if res['history']:
            fresh_history.extend(res['history'])

    if fresh_history:
        fresh_history.sort(key=lambda x: x['Time'], reverse=True)
    st.session_state.todays_history = fresh_history

    # --- UPDATED BATCH FETCH LOGIC WITH FALLBACK ---
    active_keys = list(st.session_state.active_trades.keys())
    keys_to_fetch = set()
    
    for name in active_keys:
        trade = st.session_state.active_trades[name]
        spot_key = DEFAULT_INSTRUMENTS.get(name)
        if spot_key: keys_to_fetch.add(spot_key)
        if trade['opt_key'] != "N/A": keys_to_fetch.add(trade['opt_key'])
    
    # 2. Fetch all in ONE API call
    live_quotes_map = get_live_quotes_batch(api_token, list(keys_to_fetch))
    
    # 3. Update trades from map
    for name in active_keys:
        trade = st.session_state.active_trades[name]
        spot_key = DEFAULT_INSTRUMENTS.get(name)
        if not spot_key: continue
        
        # Spot Price Update
        curr_spot = live_quotes_map.get(spot_key, 0.0)
        if curr_spot == 0: 
            # If batch failed for spot, try individual or use snapshot
            curr_spot = st.session_state.market_snapshot.get(name, {}).get('price', 0)
        
        # If still 0, fallback to entry (keeps math safe)
        if curr_spot == 0: curr_spot = float(trade.get('exit_spot', trade['entry_spot']))
        
        # Option Price Update
        curr_opt = 0.0
        if trade['opt_key'] != "N/A":
             # Try batch result first
             q = live_quotes_map.get(trade['opt_key'], 0.0)
             if q > 0: 
                 curr_opt = float(q)
             else:
                 # --- FALLBACK: Direct fetch if batch returned 0 for option ---
                 # This fixes the "stuck at entry price" issue
                 direct_q = get_live_quote(api_token, trade['opt_key'])
                 curr_opt = float(direct_q) if direct_q > 0 else float(trade['opt_entry'])

        # Logic: If curr_opt is STILL 0 (api failure), fallback to entry to avoid huge negative PnL?
        # Better to keep it as entry so PnL is 0 rather than -100%
        if curr_opt == 0: curr_opt = float(trade['opt_entry'])
            
        if trade.get('side') == 'PE' or "Sell" in trade.get('reason', ''): 
            spot_pnl = float(trade['entry_spot']) - curr_spot 
        else: 
            spot_pnl = curr_spot - float(trade['entry_spot'])
        
        opt_pnl = curr_opt - float(trade['opt_entry'])
        
        current_lot_size = int(LOT_SIZES.get(name, 25))
        pnl_inr = opt_pnl * current_lot_size * int(lot_multiplier)
        
        # Update Trade Dict
        trade['curr_opt_price'] = curr_opt
        trade['opt_pnl'] = opt_pnl
        trade['pnl_inr'] = pnl_inr
        
        if use_trailing:
            if trade.get('side') == 'CE':
                if curr_spot > trade['highest_spot']:
                    trade['highest_spot'] = curr_spot
                    if (trade['highest_spot'] - trade['entry_spot']) >= trail_start:
                        new_sl = trade['highest_spot'] - trail_offset
                        if new_sl > trade['stop_spot']:
                            trade['stop_spot'] = new_sl
                            log_message(f"{name}: Trailing SL -> {new_sl:.2f}")
                            
        exit_msg = ""; is_exit = False
        if trade.get('side') == 'CE':
            if curr_spot >= trade['target_spot']: exit_msg = "Target Hit"; is_exit = True
            elif curr_spot <= trade['stop_spot']: exit_msg = "SL Hit"; is_exit = True
        else:
            if curr_spot <= trade['target_spot']: exit_msg = "Target Hit"; is_exit = True
            elif curr_spot >= trade['stop_spot']: exit_msg = "SL Hit"; is_exit = True
            
        if is_exit:
            log_message(f"{name}: {exit_msg} | PnL: ₹{pnl_inr:.0f}", "success" if pnl_inr>0 else "error")
            st.toast(f"💰 {name}: {exit_msg}", icon="💰")
            trade.update({'exit_spot': curr_spot, 'spot_pnl': spot_pnl, 'opt_exit': curr_opt, 'opt_pnl': opt_pnl, 'pnl_inr': pnl_inr, 'exit_status': exit_msg})
            save_trade_to_csv(trade)
            del st.session_state.active_trades[name]

    st.session_state.last_scan_time = datetime.now().strftime("%H:%M:%S")

# ==========================================
# UI DASHBOARD
# ==========================================

col1, col2, col3 = st.columns([2, 4, 2])
with col1:
    status_text = "🟢 LIVE SCANNING" if auto_refresh else "⏸️ PAUSED"
    st.markdown(f"### {status_text}")
with col3:
    if st.button("🔄 Force Scan Now", use_container_width=True):
        run_parallel_scan()
        st.rerun()

st.divider()

if st.session_state.active_trades:
    st.subheader("🟢 Active Positions")
    
    view_mode = st.radio("View Mode", ["Cards", "Table"], horizontal=True, label_visibility="collapsed")
    
    if view_mode == "Cards":
        for name, t in st.session_state.active_trades.items():
            with st.container(border=True):
                c_head, c_price, c_opt, c_pnl = st.columns([2, 2, 3, 2])
                
                # Fetch live directly for rendering fallback
                inst_key = DEFAULT_INSTRUMENTS.get(name)
                
                # Dynamic Lot size for display
                ls = int(LOT_SIZES.get(name, 0))
                qty_display = f"{ls} x {lot_multiplier}"
                
                # Get Spot Price (Prioritize Snapshot -> Live Quote -> Entry)
                snap = st.session_state.market_snapshot.get(name)
                curr = snap['price'] if snap else (get_live_quote(api_token, inst_key) or t['entry_spot'])
                if curr == 0: curr = float(t['entry_spot'])
                
                # Use value from trade dict updated in scan
                curr_opt_display = t.get('curr_opt_price', t['opt_entry'])
                opt_pnl_display = t.get('opt_pnl', 0.0)
                pnl_val_display = t.get('pnl_inr', 0.0)
                
                c_head.markdown(f"#### {name}"); c_head.caption(f"{t.get('reason', 'Signal')}")
                c_price.metric("Spot Price", f"{curr:.2f}", delta=f"{curr - t['entry_spot']:.2f}")
                c_price.caption(f"Entry: {t['entry_spot']:.2f}")
                
                c_opt.metric(f"Option ({t['opt_strike']} {t.get('side','CE')})", f"{curr_opt_display:.2f}", delta=f"{opt_pnl_display:.2f} pts")
                c_opt.caption(f"Entry: {t['opt_entry']:.2f} | Qty: {qty_display}")
                
                c_pnl.metric("Total P&L", f"₹ {pnl_val_display:,.0f}", delta_color="normal", help=f"Calculation: {opt_pnl_display:.2f} * {ls} * {lot_multiplier}")
                
                range_total = abs(t['target_spot'] - t['entry_spot'])
                dist_covered = abs(curr - t['entry_spot'])
                progress = min(dist_covered / range_total, 1.0) if range_total > 0 else 0
                st.progress(progress)
                t1, t2, t3 = st.columns(3)
                t1.caption(f"🛑 SL: {t['stop_spot']:.2f}"); t2.caption(f"🎯 Target: {t['target_spot']:.2f}"); t3.caption(f"Time: {t['entry_time']}")
    else:
        table_data = []
        for name, t in st.session_state.active_trades.items():
            inst_key = DEFAULT_INSTRUMENTS.get(name)
            
            # Use snapshot or fetch
            snap = st.session_state.market_snapshot.get(name)
            curr = snap['price'] if snap else (get_live_quote(api_token, inst_key) or t['entry_spot'])
            
            table_data.append({ 
                "Instrument": name, 
                "Side": t.get('side', 'CE'), 
                "Spot": f"{curr:.2f}", 
                "Entry Spot": f"{t['entry_spot']:.2f}", 
                "Option": f"{t['opt_strike']} {t.get('side','CE')}", 
                "Opt Price": f"{t.get('curr_opt_price',0):.2f}", 
                "Opt PnL": f"{t.get('opt_pnl',0):.2f}", 
                "Total P&L (₹)": t.get('pnl_inr', 0) 
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

st.divider()

st.subheader("📊 Market Watch")
if st.session_state.market_snapshot:
    cols = st.columns(4)
    for i, (name, data) in enumerate(st.session_state.market_snapshot.items()):
        cols[i % 4].metric(name, f"{data['price']:.2f}", data['status'])

if st.session_state.todays_history:
    with st.expander("📅 Today's Signals & History", expanded=True):
        df_hist = pd.DataFrame(st.session_state.todays_history)
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True, column_config={"Profit (INR)": st.column_config.NumberColumn(format="₹ %.2f"), "Opt PnL": st.column_config.NumberColumn(format="%.2f")})

with st.expander("📜 Logs", expanded=False):
    for l in st.session_state.logs[:5]:
        icon = "✅" if l['type'] == 'success' else "❌" if l['type'] == 'error' else "ℹ️"
        st.markdown(f"`{l['time']}` {icon} {l['msg']}")

if auto_refresh and api_token:
    placeholder = st.empty()
    for i in range(refresh_rate, 0, -1):
        placeholder.markdown(f"⏳ Next scan in **{i}** seconds...")
        time.sleep(1)
    placeholder.empty()
    run_parallel_scan()
    st.rerun()

if st.session_state.last_scan_time is None and api_token:
    run_parallel_scan()
    st.rerun()
