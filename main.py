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

SESSIONS = {
    "London": (time(8,0), time(16,0)),
    "NewYork": (time(13,0), time(21,0))
}

portfolio = {}

def current_session():
    now = datetime.now(UTC).time()
    for k,(a,b) in SESSIONS.items():
        if a <= now <= b:
            return k
    return None

def news_sentiment(symbol):
    try:
        q = symbol.replace("USDT","")
        r = requests.get("https://newsapi.org/v2/everything",
                         params={"q":q,"language":"en","apiKey":NEWS_API,"pageSize":5},
                         timeout=10).json()
        score = sum(TextBlob(a.get("title","")).sentiment.polarity for a in r.get("articles",[]))
        return score / max(len(r.get("articles",[])),1)
    except: return 0

def crypto_data(sym, tf, l=300):
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol":sym,"interval":tf,"limit":l},timeout=10).json()
        df = pd.DataFrame(r, columns=list("tohlcv")+["x"]*6)
        df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].astype(float)
        return df
    except:
        return pd.DataFrame(columns=["o","h","l","c","v"])

def forex_data(pair, tf):
    try:
        r = requests.get("https://www.alphavantage.co/query",
                         params={"function":"FX_INTRADAY",
                                 "from_symbol":pair[:3],
                                 "to_symbol":pair[3:],
                                 "interval":tf,
                                 "apikey":FX_API},timeout=10).json()
        ts=[v for k,v in r.items() if "Time Series" in k]
        if not ts: return pd.DataFrame(columns=["o","h","l","c"])
        df=pd.DataFrame(ts[0]).T.astype(float)
        df.rename(columns={"1. open":"o","2. high":"h","3. low":"l","4. close":"c"}, inplace=True)
        return df
    except:
        return pd.DataFrame(columns=["o","h","l","c"])

def enrich(df):
    if len(df)==0: return df
    try:
        df["EMA20"]=ta.trend.EMAIndicator(df["c"],20).ema_indicator()
        df["EMA50"]=ta.trend.EMAIndicator(df["c"],50).ema_indicator()
        df["RSI"]=ta.momentum.RSIIndicator(df["c"],14).rsi()
        macd=ta.trend.MACD(df["c"])
        df["MACD"], df["MS"]=macd.macd(), macd.macd_signal()
        df["ATR"]=ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"],14).average_true_range()
        df["VWAP"]=(df["v"]*(df["h"]+df["l"]+df["c"])/3).cumsum()/df["v"].cumsum()
        df["StochRSI"]=ta.momentum.StochRSIIndicator(df["c"],14,3,3).stochrsi()
        df["BB_upper"]=ta.volatility.BollingerBands(df["c"],20,2).bollinger_hband()
        df["BB_lower"]=ta.volatility.BollingerBands(df["c"],20,2).bollinger_lband()
        df["PSAR"]=ta.trend.PSARIndicator(df["h"],df["l"],df["c"],0.02,0.2).psar()
        df["ADX"]=ta.trend.ADXIndicator(df["h"],df["l"],df["c"],14).adx()
        df["Ichimoku_a"]=ta.trend.IchimokuIndicator(df["h"],df["l"],9,26,52).ichimoku_a()
        df["Ichimoku_b"]=ta.trend.IchimokuIndicator(df["h"],df["l"],9,26,52).ichimoku_b()
    except:
        pass
    return df

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
        df=df.copy()
        df["r"]=df["c"].pct_change()
        df["v"]=df["r"].rolling(10).std()
        df["rsi"]=ta.momentum.RSIIndicator(df["c"],14).rsi()
        df["ema"]=ta.trend.EMAIndicator(df["c"],20).ema_indicator()
        df["ed"]=df["c"]-df["ema"]
        df=df.dropna()
        return df[["r","v","rsi","ed","c"]]

    def prepare(self, df):
        f=self.features(df)
        s=self.scaler.fit_transform(f)
        X,y=[],[]
        for i in range(self.window,len(s)-1):
            X.append(s[i-self.window:i])
            y.append(1 if f["c"].iloc[i+1]>f["c"].iloc[i] else 0)
        return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32)

    def train_daily(self, df):
        if os.path.exists(TRAIN_LOG):
            try:
                last = datetime.fromisoformat(open(TRAIN_LOG).read())
                if datetime.utcnow()-last < timedelta(hours=24): return
            except: pass
        X,y=self.prepare(df)
        if len(X)==0: return
        self.model.fit(X,y,epochs=4,batch_size=8,verbose=0)
        self.model.save(MODEL_PATH)
        open(TRAIN_LOG,"w").write(str(datetime.utcnow()))

    def predict(self, df):
        f=self.features(df)
        s=self.scaler.transform(f)
        if len(s)<self.window: return None
        X=np.array([s[-self.window:]],dtype=np.float32)
        return float(self.model.predict(X,verbose=0)[0][0])

class RLTrader:
    def decide(self, prob, news):
        score = abs(prob-0.5)*2 + abs(news)
        if score<0.75: return "NO TRADE", score
        return ("BUY" if prob>0.5 else "SELL"), score

def select_trade_style(df):
    """Automatically choose scalping or swing based on volatility"""
    if len(df)<20: return "scalp"
    atr = ta.volatility.AverageTrueRange(df["h"], df["l"], df["c"], 14).average_true_range().iloc[-1]
    recent_range = df["h"].iloc[-10:].max() - df["l"].iloc[-10:].min()
    if recent_range/df["c"].iloc[-1] > 0.01 or atr/df["c"].iloc[-1]>0.005:
        return "swing"
    return "scalp"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton(a, callback_data=a)] for a in sorted(set(CRYPTO+FOREX))]
    await update.message.reply_text("Select Asset for AI Trading", reply_markup=InlineKeyboardMarkup(kb))

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    asset = q.data
    is_crypto = asset in CRYPTO
    session = current_session()
    if not is_crypto and session is None:
        await q.edit_message_text("⏰ Forex market session closed")
        return
    await q.edit_message_text(f"🔍 Analyzing {asset}...\nPlease wait ⏳")
    try:
        df = crypto_data(asset,"5m") if is_crypto else forex_data(asset,"5min")
        df = enrich(df)
        if len(df)<100:
            await q.edit_message_text("❌ Not enough market data")
            return
        style = select_trade_style(df)
        ai = MarketAI()
        ai.train_daily(df)
        prob = ai.predict(df)
        if prob is None:
            await q.edit_message_text("❌ AI could not generate prediction")
            return
        news = news_sentiment(asset)
        rl = RLTrader()
        decision, confidence_score = rl.decide(prob, news)
        if decision=="NO TRADE":
            await q.edit_message_text("⚠️ No high-probability trade found")
            return
        price = df["c"].iloc[-1]
        atr = ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"],14).average_true_range().iloc[-1]
        if style=="scalp":
            sl = price - atr if decision=="BUY" else price + atr
            tp = price + atr if decision=="BUY" else price - atr
        else:
            sl = price - atr*1.5 if decision=="BUY" else price + atr*1.5
            tp = price + atr*3 if decision=="BUY" else price - atr*3
        portfolio[asset] = {"direction":decision,"entry":price,"SL":sl,"TP":tp,"confidence":confidence_score,"style":style}
        explanation = (
            f"🧠 AI Hedge Fund Trade Plan\n\n"
            f"Asset: {asset}\n"
            f"Style: {style}\n"
            f"Direction: {decision}\n"
            f"Entry: {round(price,2)}\n"
            f"SL: {round(sl,2)}\n"
            f"TP: {round(tp,2)}\n\n"
            f"AI Probability: {round(prob,3)}\n"
            f"News Sentiment: {round(news,3)}\n"
            f"Confidence: {round(confidence_score*100,1)}%\n\n"
            f"EMA20: {round(df['EMA20'].iloc[-1],2)}\n"
            f"EMA50: {round(df['EMA50'].iloc[-1],2)}\n"
            f"RSI: {round(df['RSI'].iloc[-1],2)}\n"
            f"MACD: {round(df['MACD'].iloc[-1],2)}"
        )
        await q.edit_message_text(explanation)
    except Exception as e:
        await q.edit_message_text(f"❌ Error during analysis:\n{str(e)}")

if __name__=="__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(analyze))
    print("Bot running")
    app.run_polling()
