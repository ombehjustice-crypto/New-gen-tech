import asyncio
import json
import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import ta
import websockets
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TOKEN = "8586615626:AAHryD22Ct8JTZZ9XgGGp9vIvdLRW-Bs0a8"
DERIV_APP_ID = "128530"
DERIV_API = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

DB_NAME = "scalper_ai.db"
MODEL_FILE = "scalper_model.pkl"

ACCOUNT_BALANCE = 1000
RISK_PER_TRADE = 0.01
CONFIDENCE_THRESHOLD = 0.65
LOSS_STREAK_LIMIT = 3
TRADE_COOLDOWN = 60  # seconds

SYMBOLS = {
    "Vol 25": "R_25",
    "Vol 50": "R_50",
    "Vol 75": "R_75",
    "Vol 100": "R_100",
    "Boom 500": "BOOM500",
}

trade_history = {}
last_trade_time = {}

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        direction TEXT,
        entry REAL,
        sl REAL,
        tp REAL,
        result INTEGER,
        confidence REAL,
        timestamp TEXT
    )""")
    conn.commit()
    conn.close()

async def fetch_data(symbol, count, granularity):
    async with websockets.connect(DERIV_API) as ws:
        req = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "style": "candles",
            "granularity": granularity,
        }
        await ws.send(json.dumps(req))
        res = json.loads(await ws.recv())
        if "error" in res:
            raise Exception(res["error"]["message"])
        df = pd.DataFrame(res["candles"])
        df["datetime"] = pd.to_datetime(df["epoch"], unit="s")
        df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
        return df

def add_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"],14).rsi()
    df["ema50"] = ta.trend.EMAIndicator(df["close"],50).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"],df["low"],df["close"],14).average_true_range()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df.dropna(inplace=True)
    return df

def equal_highs_lows(df, tol=0.0001):
    if abs(df["high"].iloc[-1] - df["high"].iloc[-2]) < tol:
        return "EQH"
    if abs(df["low"].iloc[-1] - df["low"].iloc[-2]) < tol:
        return "EQL"
    return None

def liquidity_sweep(df):
    prev_high = df["high"].iloc[-2]
    prev_low = df["low"].iloc[-2]
    close = df["close"].iloc[-1]
    if df["high"].iloc[-1] > prev_high and close < prev_high:
        return "SELL"
    if df["low"].iloc[-1] < prev_low and close > prev_low:
        return "BUY"
    return None

def break_structure(df):
    last_high = df["high"].iloc[-3:-1].max()
    last_low = df["low"].iloc[-3:-1].min()
    if df["close"].iloc[-1] > last_high:
        return "BOS_UP"
    if df["close"].iloc[-1] < last_low:
        return "BOS_DOWN"
    return None

def fair_value_gap(df):
    if df["low"].iloc[-1] > df["high"].iloc[-3]:
        return "BULL"
    if df["high"].iloc[-1] < df["low"].iloc[-3]:
        return "BEAR"
    return None

def train_model(df):
    df["target"] = (df["close"].shift(-2) > df["close"]).astype(int)
    df.dropna(inplace=True)
    X = df[["rsi","ema50","atr","macd"]]
    y = df["target"]
    if len(np.unique(y)) < 2:
        return None
    model = XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.05,eval_metric="logloss")
    model.fit(X,y)
    joblib.dump(model,MODEL_FILE)
    return model

def load_model(df):
    if not os.path.exists(MODEL_FILE):
        return train_model(df)
    return joblib.load(MODEL_FILE)

def position_size(entry, sl, confidence):
    risk_amount = ACCOUNT_BALANCE * RISK_PER_TRADE * min(confidence, 1)
    stop_distance = abs(entry - sl)
    if stop_distance == 0:
        return 0
    return risk_amount / stop_distance

def calculate_equity_curve():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM trades", conn)
    conn.close()
    df["profit"] = df.apply(lambda r: (r["tp"]-r["entry"] if r["direction"]=="BUY" else r["entry"]-r["tp"])*position_size(r["entry"], r["sl"], r["confidence"]), axis=1)
    return df["profit"].cumsum()

def record_trade(symbol, direction, entry, sl, tp, result, confidence):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO trades(symbol,direction,entry,sl,tp,result,confidence,timestamp) VALUES(?,?,?,?,?,?,?,?)",
              (symbol,direction,entry,sl,tp,result,confidence,datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    # update trade history
    if symbol not in trade_history:
        trade_history[symbol] = []
    trade_history[symbol].append(result)
    if len(trade_history[symbol])>LOSS_STREAK_LIMIT:
        trade_history[symbol].pop(0)

async def retrain_model():
    while True:
        try:
            for symbol in SYMBOLS.values():
                df = add_indicators(await fetch_data(symbol, 500, 60))
                train_model(df)
        except: pass
        await asyncio.sleep(3600)  # hourly

def generate_signal(df1, df5, df15, model, symbol):
    bias = "BUY" if df15["close"].iloc[-1] > df15["ema50"].iloc[-1] else "SELL"
    sweep = liquidity_sweep(df1)
    bos = break_structure(df1)
    fvg = fair_value_gap(df1)
    eq = equal_highs_lows(df1)
    rsi_ok = df1["rsi"].iloc[-1] > 50 if bias=="BUY" else df1["rsi"].iloc[-1] < 50
    macd_ok = df1["macd"].iloc[-1] > df1["macd_signal"].iloc[-1] if bias=="BUY" else df1["macd"].iloc[-1] < df1["macd_signal"].iloc[-1]
    confidence = model.predict_proba(df1[["rsi","ema50","atr","macd"]].iloc[-1:])[0][1] if model else 0.5

    # Confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        return None

    # Loss streak breaker
    if trade_history.get(symbol,[]).count(0) >= LOSS_STREAK_LIMIT:
        return None

    # Trade frequency control
    now = datetime.utcnow()
    if symbol in last_trade_time and (now - last_trade_time[symbol]).total_seconds() < TRADE_COOLDOWN:
        return None

    direction = bias
    entry = df1["close"].iloc[-1]
    atr = df1["atr"].iloc[-1]
    sl = entry - atr*1.2 if direction=="BUY" else entry + atr*1.2
    tp = entry + atr*2.5 if direction=="BUY" else entry - atr*2.5
    size = position_size(entry, sl, confidence)
    debug_info = f"Sweep: {sweep}, BOS: {bos}, FVG: {fvg}, EQ: {eq}, RSI_OK: {rsi_ok}, MACD_OK: {macd_ok}"
    last_trade_time[symbol] = now
    return direction, entry, sl, tp, size, round(confidence*100,2), debug_info

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(name, callback_data=name)] for name in SYMBOLS]
    await update.message.reply_text("Synthetic Scalper AI with Adaptive Learning",reply_markup=InlineKeyboardMarkup(keyboard))

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    symbol = SYMBOLS[query.data]
    try:
        df1 = add_indicators(await fetch_data(symbol,300,60))
        df5 = add_indicators(await fetch_data(symbol,300,300))
        df15 = add_indicators(await fetch_data(symbol,300,900))
        model = load_model(df1)
        result = generate_signal(df1,df5,df15,model, symbol)
        if not result:
            await query.edit_message_text("No scalping setup or confidence below threshold / loss streak active")
            return
        direction, entry, sl, tp, size, confidence, debug_info = result
        text = (f"{query.data}\nDirection: {direction}\nEntry: {entry:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f}"
                f"\nLot: {size:.4f}\nConfidence: {confidence}%\n{debug_info}")
        await query.edit_message_text(text)
        # after-trade self-learning placeholder (simulate result as 1 for demo)
        record_trade(symbol, direction, entry, sl, tp, 1, confidence)
    except Exception as e:
        await query.edit_message_text(str(e))

def main():
    init_db()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(CallbackQueryHandler(button))
    loop = asyncio.get_event_loop()
    loop.create_task(retrain_model())
    app.run_polling()

if __name__ == "__main__":
    main()