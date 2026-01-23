import os, requests, numpy as np, pandas as pd, ta, pytz
from datetime import datetime, time, timedelta
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

TOKEN = "8597020255:AAF20Lvuy1fLBTU7h1CUYOXqOnCyfzLUTFA"
NEWS_API = "332bf45035354091b59f1f64601e2e11"
FX_API = "ca1acbf0cedb4488b130c59252891c5e"


MODEL_PATH = "ai_model_portfolio.h5"
TRAIN_LOG = "last_train_portfolio.txt"

CRYPTO = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT"]
FOREX = ["EURUSD","GBPUSD","USDJPY","XAUUSD"]
UTC = pytz.UTC

portfolio = {}

def news_sentiment(symbol):
    try:
        q = symbol.replace("USDT","")
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": q, "language": "en", "apiKey": NEWS_API,"pageSize":5},
            timeout=10
        ).json()
        articles = r.get("articles",[])
        if not articles:
            return 0
        score = sum(TextBlob(a.get("title","")).sentiment.polarity for a in articles)
        return score / len(articles)
    except:
        return 0

def crypto_data(sym, tf, l=1000):
    """Fetch Binance klines with fallback"""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol":sym,"interval":tf,"limit":l},
            timeout=10
        ).json()
        if isinstance(r, dict):
            return pd.DataFrame(columns=["o","h","l","c","v"])
        df = pd.DataFrame(r, columns=list("tohlcv")+["x"]*6)
        df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].astype(float)
        return df
    except:
        return pd.DataFrame(columns=["o","h","l","c","v"])

def forex_data(pair, tf):
    """Fetch Alpha Vantage FX with retry and fallback intervals"""
    intervals = ["1min","5min","15min","60min"]
    for interval in intervals:
        for attempt in range(3):
            try:
                r = requests.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function":"FX_INTRADAY",
                        "from_symbol":pair[:3],
                        "to_symbol":pair[3:],
                        "interval":interval,
                        "apikey":FX_API,
                        "outputsize":"full"
                    },
                    timeout=10
                ).json()
                ts = [v for k,v in r.items() if "Time Series" in k]
                if ts:
                    df = pd.DataFrame(ts[0]).T.astype(float)
                    df.rename(columns={"1. open":"o","2. high":"h","3. low":"l","4. close":"c"}, inplace=True)
                    return df.sort_index()
            except:
                continue
    return pd.DataFrame(columns=["o","h","l","c"])

def enrich(df):
    if len(df) < 20:
        return df
    df["EMA20"] = ta.trend.EMAIndicator(df["c"],20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(df["c"],50).ema_indicator()
    df["RSI"] = ta.momentum.RSIIndicator(df["c"],14).rsi()
    macd = ta.trend.MACD(df["c"])
    df["MACD"] = macd.macd()
    df["MS"] = macd.macd_signal()
    df["ATR"] = ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"],14).average_true_range()
    return df.dropna()

class MarketAI:
    def __init__(self, window=30):
        self.window = window
        self.scaler = MinMaxScaler()
        self.model = self.load_or_create()

    def load_or_create(self):
        if os.path.exists(MODEL_PATH):
            try: return load_model(MODEL_PATH)
            except: os.remove(MODEL_PATH)
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.window,5)),
            Dropout(0.2),
            LSTM(32),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def features(self, df):
        df = df.copy()
        df["r"] = df["c"].pct_change()
        df["v"] = df["r"].rolling(10).std()
        df["rsi"] = ta.momentum.RSIIndicator(df["c"],14).rsi()
        df["ema"] = ta.trend.EMAIndicator(df["c"],20).ema_indicator()
        df["ed"] = df["c"] - df["ema"]
        df = df.dropna()
        return df[["r","v","rsi","ed","c"]]

    def prepare(self, df):
        f = self.features(df)
        if len(f) <= self.window:
            return None, None
        s = self.scaler.fit_transform(f)
        X, y = [], []
        for i in range(self.window, len(s)-1):
            X.append(s[i-self.window:i])
            y.append(1 if f["c"].iloc[i+1] > f["c"].iloc[i] else 0)
        return np.array(X), np.array(y)

    def train_daily(self, df):
        X, y = self.prepare(df)
        if X is None or len(X)==0: return
        self.model.fit(X, y, epochs=3, batch_size=8, verbose=0)
        self.model.save(MODEL_PATH)

    def predict(self, df):
        f = self.features(df)
        if len(f) < self.window: return None
        s = self.scaler.fit_transform(f)
        X = np.array([s[-self.window:]])
        return float(self.model.predict(X, verbose=0)[0][0])

class RLTrader:
    def decide(self, prob, news):
        score = abs(prob-0.5)*2 + abs(news)
        if score < 0.6: return "NO TRADE", score
        return ("BUY" if prob>0.5 else "SELL"), score

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton(a, callback_data=a)] for a in sorted(set(CRYPTO+FOREX))]
    await update.message.reply_text("Select Asset for AI Trading", reply_markup=InlineKeyboardMarkup(kb))

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    asset = q.data
    is_crypto = asset in CRYPTO

    await q.edit_message_text(f"🔍 Analyzing {asset}...\nPlease wait ⏳")

    
    df = crypto_data(asset,"5m") if is_crypto else forex_data(asset,"5min")
    df = enrich(df)

    if len(df) < 30:
        
        df = crypto_data(asset,"5m") if is_crypto else forex_data(asset,"5min")
        df = enrich(df)
        if len(df) < 30:
            await q.edit_message_text("❌ Market data still loading, try again shortly")
            return

    ai = MarketAI()
    ai.train_daily(df)
    prob = ai.predict(df)
    if prob is None:
        await q.edit_message_text("❌ Analysis incomplete, retry")
        return

    news = news_sentiment(asset)
    decision, confidence = RLTrader().decide(prob, news)
    if decision == "NO TRADE":
        await q.edit_message_text("⚠️ No high-probability trade found")
        return

    price = df["c"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    sl = price - atr if decision=="BUY" else price + atr
    tp = price + atr*2 if decision=="BUY" else price - atr*2

    await q.edit_message_text(
        f"🧠 AI Hedge Fund Trade Plan\n\n"
        f"Asset: {asset}\n"
        f"Direction: {decision}\n"
        f"Entry: {round(price,5)}\n"
        f"SL: {round(sl,5)}\n"
        f"TP: {round(tp,5)}\n\n"
        f"Probability: {round(prob,3)}\n"
        f"Confidence: {round(confidence*100,1)}%"
    )

if __name__=="__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(analyze))
    print("Bot running")
    app.run_polling()