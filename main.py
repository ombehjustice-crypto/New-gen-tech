import asyncio
import json
import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import ta
import websockets
from xgboost import XGBClassifier
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TOKEN = "8586615626:AAHryD22Ct8JTZZ9XgGGp9vIvdLRW-Bs0a8"
DERIV_APP_ID = "128530"
DERIV_API =f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"


DB_NAME = "scalper_ai.db"
MODEL_FILE = "scalper_model.pkl"

ACCOUNT_BALANCE = 1000
RISK_PER_TRADE = 0.01

SYMBOLS = {
    "Vol 25": "R_25",
    "Vol 50": "R_50",
    "Vol 75": "R_75",
    "Vol 100": "R_100",
    "Boom 500": "BOOM500",
}

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
        result INTEGER
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

def position_size(entry, sl):
    risk_amount = ACCOUNT_BALANCE * RISK_PER_TRADE
    stop_distance = abs(entry - sl)
    if stop_distance == 0:
        return 0
    return risk_amount / stop_distance

def generate_signal(df1, df5, df15, model):
    bias = "BUY" if df15["close"].iloc[-1] > df15["ema50"].iloc[-1] else "SELL"
    sweep = liquidity_sweep(df1)
    bos = break_structure(df1)
    fvg = fair_value_gap(df1)
    eq = equal_highs_lows(df1)
    rsi_ok = df1["rsi"].iloc[-1] > 50 if bias=="BUY" else df1["rsi"].iloc[-1] < 50
    macd_ok = df1["macd"].iloc[-1] > df1["macd_signal"].iloc[-1] if bias=="BUY" else df1["macd"].iloc[-1] < df1["macd_signal"].iloc[-1]
    confidence = model.predict_proba(df1[["rsi","ema50","atr","macd"]].iloc[-1:])[0][1] if model else 0.5
    direction = None
    if sweep and bos and fvg and rsi_ok and macd_ok:
        direction = bias
    if not direction:
        return None
    entry = df1["close"].iloc[-1]
    atr = df1["atr"].iloc[-1]
    sl = entry - atr*1.2 if direction=="BUY" else entry + atr*1.2
    tp = entry + atr*2.5 if direction=="BUY" else entry - atr*2.5
    size = position_size(entry, sl)
    return direction, entry, sl, tp, size, round(confidence*100,2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(name, callback_data=name)] for name in SYMBOLS]
    await update.message.reply_text("Synthetic Scalper AI",reply_markup=InlineKeyboardMarkup(keyboard))

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    symbol = SYMBOLS[query.data]
    try:
        df1 = add_indicators(await fetch_data(symbol,300,60))
        df5 = add_indicators(await fetch_data(symbol,300,300))
        df15 = add_indicators(await fetch_data(symbol,300,900))
        model = load_model(df1)
        signal = generate_signal(df1,df5,df15,model)
        if not signal:
            await query.edit_message_text("No scalping setup")
            return
        direction, entry, sl, tp, size, confidence = signal
        text = f"{query.data}\nDirection: {direction}\nEntry: {entry:.2f}\nSL: {sl:.2f}\nTP: {tp:.2f}\nLot: {size:.4f}\nConfidence: {confidence}%"
        await query.edit_message_text(text)
    except Exception as e:
        await query.edit_message_text(str(e))

def main():
    init_db()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",start))
    app.add_handler(CallbackQueryHandler(button))
    app.run_polling()

if __name__ == "__main__":
    main()