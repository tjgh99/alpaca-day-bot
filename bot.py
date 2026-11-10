"""Intraday momentum/breakout bot for an Alpaca PAPER account.

Usage:
    python bot.py trade     # run the trading loop until the flatten time
    python bot.py flatten   # cancel all orders and close all positions
    python bot.py summary   # post the day's trades and P&L (skips non-trading days)

Long-only. Entries are bracket orders (server-side stop and target).
Always connects to the paper endpoint, regardless of environment settings.
"""
import json
import logging
import math
import os
import re
import smtplib
import sys
import time
import urllib.request
from datetime import datetime, timezone, time as dtime, timedelta
from email.message import EmailMessage

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "META", "GOOGL",
    "PLTR", "J", "MU", "AVGO", "NFLX", "COIN", "HOOD", "SOFI", "UBER", "SMCI",
    "CRWD", "ARM", "SHOP", "INTC", "ORCL",
]

# Headlines matching any of these block a buy and trigger an early exit on a held position.
NEGATIVE_NEWS = re.compile(
    r"\b(downgrad\w*|lawsuit|sued|sues|investigation|probe|subpoena|fraud|recall|bankruptcy|"
    r"delist\w*|halted|short seller|short report|data breach|cyberattack|resigns?|steps down|"
    r"misses|missed estimates|cuts? (?:guidance|forecast|outlook)|lowers? (?:guidance|forecast|outlook)|"
    r"(?:public|secondary|stock) offering|plunges?|plummets?|tumbles?|sinks?)\b",
    re.IGNORECASE,
)
NEWS_LOOKBACK = timedelta(hours=18)
NEWS_UNAVAILABLE = "news unavailable"
MAX_TAGGED_SYMBOLS = 3
RECENT_WINDOW = timedelta(hours=2)
RECENT_THRESHOLD = 0.70   # negative-probability needed to block when the article is under 2h old
OLDER_THRESHOLD = 0.85    # ... and when it is older

# Risk limits (moderate-high profile)
MAX_POSITION_PCT = 0.25     # max notional per position, share of equity
RISK_PER_TRADE_PCT = 0.025  # equity risked per trade (entry to stop)
DAILY_LOSS_HALT_PCT = 0.06  # stop trading and flatten at this drawdown on the day
MAX_OPEN_POSITIONS = 8
MAX_ENTRIES_PER_DAY = 15
MAX_ENTRIES_PER_SYMBOL = 2
REWARD_RISK = 2.0           # take-profit distance as a multiple of stop distance

# Signal parameters (1-minute bars)
BREAKOUT_LOOKBACK = 30
VOLUME_LOOKBACK = 20
VOLUME_MULT = 1.5
ATR_LOOKBACK = 14
ATR_STOP_MULT = 1.5
MIN_STOP_PCT = 0.004
COOLDOWN = timedelta(minutes=30)

# Entries only between these times (ET): skips the noisy open and the late-day drift
ENTRY_START = dtime(9, 45)
ENTRY_END = dtime(14, 30)

POLL_SECONDS = 30
FLATTEN_BEFORE_CLOSE = timedelta(minutes=45)
MAX_WAIT_FOR_OPEN = timedelta(minutes=20)

log = logging.getLogger("bot")


def make_clients():
    key, secret = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    return TradingClient(key, secret, paper=True), StockHistoricalDataClient(key, secret)


def make_news_client():
    return NewsClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


_finbert = None
_scores = {}
NAMES = {}  # symbol -> first word of company name, used to check an article is about the stock

# Serious events block a buy whatever the sentiment model says (it underrates legal/regulatory news).
SEVERE_NEWS = re.compile(
    r"\b(investigation|subpoena|fraud|bankruptcy|delist\w*|halted|recall|sec charges|"
    r"data breach|cyberattack|short report|short seller)\b",
    re.IGNORECASE,
)


def about_symbol(item, sym):
    """True if the article is plausibly about sym, not just tagged with it."""
    if len(item.symbols) == 1:
        return True
    text = f"{item.headline} {item.summary or ''}".lower()
    return bool(re.search(rf"\b{re.escape(sym.lower())}\b", text) or NAMES.get(sym, sym.lower()) in text)


def load_finbert():
    """Load the FinBERT sentiment model; on failure the keyword filter is used instead."""
    global _finbert
    try:
        from transformers import pipeline

        _finbert = pipeline("text-classification", model="ProsusAI/finbert", top_k=None,
                            truncation=True, max_length=256)
        log.info("FinBERT loaded")
    except Exception as exc:
        log.warning("FinBERT unavailable (%s); falling back to keyword filter", exc)


def negative_score(item):
    """Probability (0-1) that an article is negative news. Cached per article."""
    if item.id in _scores:
        return _scores[item.id]
    if _finbert is None:
        score = 1.0 if NEGATIVE_NEWS.search(item.headline) else 0.0
    else:
        out = _finbert(f"{item.headline}. {item.summary or ''}"[:1000])
        if out and isinstance(out[0], list):
            out = out[0]
        score = next(d["score"] for d in out if d["label"].lower() == "negative")
    _scores[item.id] = score
    return score


def bad_news(news, sym, since):
    """Return the most damning negative headline about sym published after `since`, or None.

    Only articles that tag few tickers count, since broad market stories say little about one stock.
    Articles from the last two hours use a lower threshold than older ones.
    Fails closed: if news can't be fetched, return a blocking marker so no buy goes out blind.
    """
    try:
        items = news.get_news(NewsRequest(symbols=sym, start=since, limit=20, sort="desc")).data["news"]
    except Exception as exc:
        log.warning("news fetch failed for %s (%s); treating as blocked", sym, exc)
        return NEWS_UNAVAILABLE
    now = datetime.now(timezone.utc)
    worst = None
    for item in items:
        if len(item.symbols) > MAX_TAGGED_SYMBOLS or not about_symbol(item, sym):
            continue
        score = 1.0 if SEVERE_NEWS.search(item.headline) else negative_score(item)
        threshold = RECENT_THRESHOLD if now - item.created_at < RECENT_WINDOW else OLDER_THRESHOLD
        if score >= threshold and (worst is None or score > worst[0]):
            worst = (score, item.headline)
    return f"{worst[1]} (negative {worst[0]:.2f})" if worst else None


def tradable_universe(trading):
    ok = []
    for sym in UNIVERSE:
        try:
            asset = trading.get_asset(sym)
            if asset.tradable:
                ok.append(sym)
                first = re.sub(r"[^a-z0-9]", "", (asset.name or "").split(" ")[0].lower())
                NAMES[sym] = first if len(first) >= 4 else sym.lower()
                continue
        except Exception:
            pass
        log.warning("dropping %s: not found or not tradable", sym)
    return ok


def flatten(trading):
    trading.cancel_orders()
    trading.close_all_positions(cancel_orders=True)
    log.info("flattened: all orders cancelled, all positions closed")


def fetch_bars(data, symbols, start):
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        feed=DataFeed.IEX,
    )
    df = data.get_stock_bars(req).df
    if df.empty:
        return {}
    return {sym: g.droplevel("symbol").sort_index() for sym, g in df.groupby(level="symbol")}


def signal(df, now):
    """Return (price, stop_distance) if the latest bar is a breakout entry, else None."""
    if len(df) < BREAKOUT_LOOKBACK + 1:
        return None
    if now - df.index[-1].to_pydatetime() > timedelta(minutes=3):
        return None  # stale data
    last = df.iloc[-1]
    prior_high = df["high"].iloc[-BREAKOUT_LOOKBACK - 1:-1].max()
    avg_vol = df["volume"].iloc[-VOLUME_LOOKBACK - 1:-1].mean()
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical * df["volume"]).sum() / df["volume"].sum()
    atr = (df["high"] - df["low"]).iloc[-ATR_LOOKBACK:].mean()
    price = float(last["close"])
    if price > prior_high and last["volume"] > VOLUME_MULT * avg_vol and price > vwap:
        return price, max(ATR_STOP_MULT * float(atr), MIN_STOP_PCT * price)
    return None


def market_is_up(spy):
    """Only take long breakouts while SPY trades above its own VWAP."""
    if spy is None or spy["volume"].sum() == 0:
        return False
    typical = (spy["high"] + spy["low"] + spy["close"]) / 3
    return float(spy["close"].iloc[-1]) > float((typical * spy["volume"]).sum() / spy["volume"].sum())


def position_size(equity, buying_power, price, stop_dist):
    qty = min(
        equity * RISK_PER_TRADE_PCT / stop_dist,
        equity * MAX_POSITION_PCT / price,
        buying_power / price,
    )
    return math.floor(qty)


def trade():
    trading, data = make_clients()
    clock = trading.get_clock()
    if not clock.is_open:
        if clock.next_open - clock.timestamp > MAX_WAIT_FOR_OPEN:
            log.info("market closed, next open %s; exiting", clock.next_open)
            return
        log.info("waiting for open at %s", clock.next_open)
        time.sleep(max(0, (clock.next_open - clock.timestamp).total_seconds()) + 5)
        clock = trading.get_clock()

    flatten_at = clock.next_close - FLATTEN_BEFORE_CLOSE
    if clock.timestamp >= flatten_at:
        log.info("past flatten time %s; exiting", flatten_at)
        return

    day_start = clock.timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
    last_entry = {}
    entries = {}
    last_market_up = None
    loops = 0
    news = make_news_client()
    load_finbert()
    symbols = tradable_universe(trading)
    started = clock.timestamp
    log.info("trading %d symbols until %s", len(symbols), flatten_at)

    while True:
        now = trading.get_clock().timestamp
        if now >= flatten_at:
            flatten(trading)
            return

        acct = trading.get_account()
        equity, last_equity = float(acct.equity), float(acct.last_equity)
        if acct.trading_blocked:
            log.error("trading blocked on account; exiting")
            return
        if equity / last_equity - 1 <= -DAILY_LOSS_HALT_PCT:
            log.warning("daily loss limit hit (equity %.2f vs %.2f); halting", equity, last_equity)
            flatten(trading)
            return

        held = {p.symbol for p in trading.get_all_positions()}

        # Early exit: negative headline published since the bot started
        for sym in sorted(held):
            headline = bad_news(news, sym, started)
            if headline and headline != NEWS_UNAVAILABLE:
                for o in trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[sym])):
                    trading.cancel_order_by_id(o.id)
                trading.close_position(sym)
                log.info("SELL %s early on news: %s", sym, headline)
                held.discard(sym)

        pending = {
            o.symbol
            for o in trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
        }
        slots = MAX_OPEN_POSITIONS - len(held)
        in_window = ENTRY_START <= now.time() <= ENTRY_END

        if slots > 0 and in_window and sum(entries.values()) < MAX_ENTRIES_PER_DAY:
            bars = fetch_bars(data, symbols, day_start)
            market_up = market_is_up(bars.get("SPY"))
            if market_up != last_market_up or loops % 20 == 0:
                log.info("status: SPY above VWAP=%s, positions=%d, entries today=%d, equity=%.2f",
                         market_up, len(held), sum(entries.values()), equity)
                last_market_up = market_up
            loops += 1
            if not market_up:
                time.sleep(POLL_SECONDS)
                continue
            buying_power = float(acct.buying_power)
            for sym in symbols:
                if slots <= 0:
                    break
                if sym in held or sym in pending or sym not in bars:
                    continue
                if entries.get(sym, 0) >= MAX_ENTRIES_PER_SYMBOL:
                    continue
                if sym in last_entry and now - last_entry[sym] < COOLDOWN:
                    continue
                sig = signal(bars[sym], now)
                if not sig:
                    continue
                price, stop_dist = sig
                headline = bad_news(news, sym, now - NEWS_LOOKBACK)
                if headline:
                    log.info("skip %s breakout, news: %s", sym, headline)
                    last_entry[sym] = now  # don't re-check this symbol every 30s
                    continue
                qty = position_size(equity, buying_power, price, stop_dist)
                if qty < 1:
                    continue
                target = round(price + REWARD_RISK * stop_dist, 2)
                stop = round(price - stop_dist, 2)
                trading.submit_order(
                    MarketOrderRequest(
                        symbol=sym,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                        order_class=OrderClass.BRACKET,
                        take_profit=TakeProfitRequest(limit_price=target),
                        stop_loss=StopLossRequest(stop_price=stop),
                    )
                )
                log.info("BUY %s x%d @~%.2f stop %.2f target %.2f", sym, qty, price, stop, target)
                last_entry[sym] = now
                entries[sym] = entries.get(sym, 0) + 1
                buying_power -= qty * price
                slots -= 1

        time.sleep(POLL_SECONDS)


def summary():
    trading, _ = make_clients()
    clock = trading.get_clock()
    today = clock.timestamp.date()
    if not trading.get_calendar(GetCalendarRequest(start=today, end=today)):
        log.info("%s is not a trading day; no summary", today)
        return
    acct = trading.get_account()
    equity, last_equity = float(acct.equity), float(acct.last_equity)
    after = clock.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    orders = trading.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=after, limit=500)
    )
    fills = sorted(
        (o for o in orders if o.filled_qty and float(o.filled_qty) > 0),
        key=lambda o: o.filled_at,
    )
    lines = [
        f"Paper account summary for {today}",
        f"Equity: ${equity:,.2f}  ({equity - last_equity:+,.2f}, {equity / last_equity - 1:+.2%} vs prior close)",
        f"Buying power: ${float(acct.buying_power):,.2f}",
        "",
        f"Fills today: {len(fills)}",
    ]
    for o in fills:
        t = o.filled_at.astimezone(clock.timestamp.tzinfo).strftime("%H:%M")
        lines.append(f"  {t}  {o.side.value.upper():4} {o.symbol:5} x{o.filled_qty} @ {float(o.filled_avg_price):.2f}")
    positions = trading.get_all_positions()
    lines += ["", f"Open positions: {len(positions)}"]
    for p in positions:
        lines.append(f"  {p.symbol} x{p.qty}  unrealized {float(p.unrealized_pl):+,.2f}")

    subject = f"Trading bot {today}: {equity - last_equity:+,.2f} ({equity / last_equity - 1:+.2%})"
    body = "\n".join(lines)
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        payload = json.dumps({"text": f"*{subject}*\n```{body}```"}).encode()
        req = urllib.request.Request(webhook, payload, {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
        log.info("summary posted to Slack")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = os.environ["EMAIL_TO"]
    msg.set_content(body)
    with smtplib.SMTP("smtp-mail.outlook.com", 587) as smtp:
        smtp.starttls()
        smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)
    log.info("summary emailed to %s", msg["To"])


def check():
    """Read-only diagnostic: verify keys, ticker list and news access without trading."""
    trading, data = make_clients()
    symbols = tradable_universe(trading)
    log.info("tradable: %d/%d symbols; dropped: %s", len(symbols), len(UNIVERSE),
             sorted(set(UNIVERSE) - set(symbols)))
    news = make_news_client()
    load_finbert()
    if _finbert:
        samples = [
            "Micron beats earnings estimates and raises guidance",
            "SEC opens investigation into company accounting practices",
            "Probe cleared, shares jump as regulator drops case",
            "Analyst downgrades stock citing slowing demand",
            "Company announces new product launch",
        ]
        for text in samples:
            out = _finbert(text)
            out = out[0] if out and isinstance(out[0], list) else out
            log.info("sentiment %s -> %s", text, {d["label"]: round(d["score"], 2) for d in out})
    since = trading.get_clock().timestamp - NEWS_LOOKBACK
    items = news.get_news(NewsRequest(symbols="NVDA", start=since, limit=5, sort="desc")).data["news"]
    log.info("news ok: %d items for NVDA; sample: %s", len(items), items[0].headline if items else "none")
    for sym in symbols:
        log.info("%s news block: %s", sym, bad_news(news, sym, since))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mode = sys.argv[1] if len(sys.argv) > 1 else "trade"
    if mode == "flatten":
        flatten(make_clients()[0])
    elif mode == "trade":
        trade()
    elif mode == "summary":
        summary()
    elif mode == "check":
        check()
    else:
        sys.exit(f"unknown mode {mode!r}; use trade, flatten or summary")
