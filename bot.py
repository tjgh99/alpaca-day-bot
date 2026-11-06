"""Intraday momentum/breakout bot for an Alpaca PAPER account.

Usage:
    python bot.py trade     # run the trading loop until the flatten time
    python bot.py flatten   # cancel all orders and close all positions
    python bot.py summary   # email the day's trades and P&L (skips non-trading days)

Long-only. Entries are bracket orders (server-side stop and target).
Always connects to the paper endpoint, regardless of environment settings.
"""
import logging
import math
import os
import smtplib
import sys
import time
from datetime import timedelta
from email.message import EmailMessage

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
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

UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "META", "GOOGL"]

# Risk limits (moderate profile)
MAX_POSITION_PCT = 0.20     # max notional per position, share of equity
RISK_PER_TRADE_PCT = 0.02   # equity risked per trade (entry to stop)
DAILY_LOSS_HALT_PCT = 0.05  # stop trading and flatten at this drawdown on the day
MAX_OPEN_POSITIONS = 5
REWARD_RISK = 2.0           # take-profit distance as a multiple of stop distance

# Signal parameters (1-minute bars)
BREAKOUT_LOOKBACK = 30
VOLUME_LOOKBACK = 20
VOLUME_MULT = 1.5
ATR_LOOKBACK = 14
ATR_STOP_MULT = 1.5
MIN_STOP_PCT = 0.004
COOLDOWN = timedelta(minutes=15)

POLL_SECONDS = 30
FLATTEN_BEFORE_CLOSE = timedelta(minutes=45)
MAX_WAIT_FOR_OPEN = timedelta(minutes=20)

log = logging.getLogger("bot")


def make_clients():
    key, secret = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    return TradingClient(key, secret, paper=True), StockHistoricalDataClient(key, secret)


def flatten(trading):
    trading.cancel_orders()
    trading.close_all_positions(cancel_orders=True)
    log.info("flattened: all orders cancelled, all positions closed")


def fetch_bars(data, start):
    req = StockBarsRequest(
        symbol_or_symbols=UNIVERSE,
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
    log.info("trading until %s", flatten_at)

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
        pending = {
            o.symbol
            for o in trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
        }
        slots = MAX_OPEN_POSITIONS - len(held)

        if slots > 0:
            bars = fetch_bars(data, day_start)
            buying_power = float(acct.buying_power)
            for sym in UNIVERSE:
                if slots <= 0:
                    break
                if sym in held or sym in pending or sym not in bars:
                    continue
                if sym in last_entry and now - last_entry[sym] < COOLDOWN:
                    continue
                sig = signal(bars[sym], now)
                if not sig:
                    continue
                price, stop_dist = sig
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

    msg = EmailMessage()
    msg["Subject"] = f"Trading bot {today}: {equity - last_equity:+,.2f} ({equity / last_equity - 1:+.2%})"
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = os.environ["EMAIL_TO"]
    msg.set_content("\n".join(lines))
    with smtplib.SMTP("smtp-mail.outlook.com", 587) as smtp:
        smtp.starttls()
        smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)
    log.info("summary emailed to %s", msg["To"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mode = sys.argv[1] if len(sys.argv) > 1 else "trade"
    if mode == "flatten":
        flatten(make_clients()[0])
    elif mode == "trade":
        trade()
    elif mode == "summary":
        summary()
    else:
        sys.exit(f"unknown mode {mode!r}; use trade, flatten or summary")
