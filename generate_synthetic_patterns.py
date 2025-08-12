"""
generate_synthetic_patterns.py

Generates synthetic moonshot and non-moonshot token patterns for training the analyzer.

Outputs:
  data/moonshot_patterns.json  (20,000 records by default: 10k good, 10k bad)

Usage:
  python generate_synthetic_patterns.py
"""

import os
import json
import math
import random
from typing import Dict, List
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# -----------------------
# CONFIG
# -----------------------
OUT_DIR = "data"
OUT_FILE = os.path.join(OUT_DIR, "moonshot_patterns.json")
N_GOOD = 10000
N_BAD = 10000
RND_SEED = 42

# Set reproducible randomness
random.seed(RND_SEED)
np.random.seed(RND_SEED)

# -----------------------
# UTILITIES
# -----------------------
def geom_price_path(initial_price: float, minutes: int, mu: float, sigma: float) -> List[float]:
    """
    Geometric Brownian Motion price path sampled each minute.
    mu: drift per minute (e.g. 0.01 means +1% expected per minute)
    sigma: volatility per minute
    Returns list of length `minutes+1` including initial price.
    """
    prices = [initial_price]
    for _ in range(minutes):
        dt = 1.0
        # dS = mu * S * dt + sigma * S * sqrt(dt) * Z
        z = np.random.normal()
        S_prev = prices[-1]
        S_new = S_prev * math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)
        prices.append(max(S_new, 1e-9))
    return prices

def sample_holder_counts(base_holders: int, minutes: int, growth_rate: float):
    """
    Simulate holder counts across minutes with Poisson arrivals scaled by growth_rate.
    """
    holders = [base_holders]
    for t in range(1, minutes+1):
        new = holders[-1] + np.random.poisson(max(1, growth_rate * holders[-1] * 0.01))
        holders.append(int(new))
    return holders

def wallet_concentration_top10(base_conc: float, minutes: int, decay: float):
    """
    Simulate top10 concentration proportion over time (0..1). decay is per-minute relative change.
    """
    vals = [base_conc]
    for _ in range(minutes):
        noise = np.random.normal(scale=0.005)
        new = vals[-1] * (1 - decay) + noise
        new = min(max(new, 0.01), 0.99)
        vals.append(new)
    return vals

def simulate_transaction_counts(volume_series: List[float], aggressiveness: float):
    """
    Map volume to buy/sell counts: higher volume => more Txns.
    aggressiveness scales buys relative to sells for good tokens.
    Returns buys_per_minute, sells_per_minute lists.
    """
    buys = []
    sells = []
    for v in volume_series:
        # base rate proportional to log volume
        base = max(1, int(math.log1p(v + 1) * 3))
        # randomness
        b = np.random.poisson(base * max(0.2, aggressiveness))
        s = np.random.poisson(base * max(0.05, 1 - aggressiveness))
        buys.append(int(b))
        sells.append(int(s))
    return buys, sells

def aggregate_window(series, start_min, window_min):
    """
    Aggregate an array sampled each minute: returns sum over [start_min, start_min+window_min)
    """
    arr = np.array(series, dtype=float)
    return float(arr[start_min:start_min+window_min].sum())

# -----------------------
# SEED TEMPLATES (small set of archetypes)
# -----------------------
# These seeds are *heuristic* archetypes
GOOD_SEEDS = [
    {
        "name": "VINE-like",
        "initial_price": 0.001,
        "initial_liquidity": 200_000,
        "initial_marketcap": 2_000_000,
        "base_holders": 300,
        "base_conc": 0.18,   # top10 concentration
        "aggressiveness": 1.4,  # buyer-heavy
    },
    {
        "name": "TRUMP-like",
        "initial_price": 0.005,
        "initial_liquidity": 150_000,
        "initial_marketcap": 1_500_000,
        "base_holders": 250,
        "base_conc": 0.20,
        "aggressiveness": 1.2,
    }
]

BAD_SEEDS = [
    {
        "name": "low-interest",
        "initial_price": 0.0004,
        "initial_liquidity": 20_000,
        "initial_marketcap": 25_000,
        "base_holders": 30,
        "base_conc": 0.6,
        "aggressiveness": 0.4
    },
    {
        "name": "bot-pump-dump",
        "initial_price": 0.0008,
        "initial_liquidity": 80_000,
        "initial_marketcap": 80_000,
        "base_holders": 80,
        "base_conc": 0.45,
        "aggressiveness": 0.6
    }
]

# -----------------------
# MAIN PATTERN GENERATOR
# -----------------------
def generate_single_pattern(seed: Dict, label: str) -> Dict:
    """
    Generate one pattern sampled from a seed archetype.
    label: "good" or "bad"
    """
    # define dynamics depending on label
    if label == "good":
        # quick liquidity add, high upward drift early, moderate vol
        liquidity_added_at_minute = int(np.random.choice([0,1,2,3,4], p=[0.5,0.2,0.15,0.1,0.05]))
        mu_early = np.random.uniform(0.06, 0.25)   # +6% to +25% per minute early (aggressive)
        sigma_early = np.random.uniform(0.12, 0.35)
        holder_growth = np.random.uniform(0.01, 0.05)
        conc_decay = np.random.uniform(0.01, 0.06)
        aggressiveness = min(seed.get("aggressiveness",1.0), 2.0)
    else:
        # bad: slow/no meaningful growth, high concentration, sells likely
        liquidity_added_at_minute = int(np.random.choice([0,1,2,3,4,5,10], p=[0.2,0.15,0.15,0.1,0.1,0.1,0.2]))
        mu_early = np.random.uniform(-0.02, 0.03)  # flat or slightly up, often no real pump
        sigma_early = np.random.uniform(0.05, 0.25)
        holder_growth = np.random.uniform(-0.01, 0.02)
        conc_decay = np.random.uniform(-0.01, 0.02)
        aggressiveness = seed.get("aggressiveness", 0.5)

    minutes = 60  # simulate 60 minutes (we will aggregate to 10/30/60 windows)
    # initial conditions
    init_price = seed["initial_price"] * (1 + np.random.normal(scale=0.2))
    init_liquidity = max(1000.0, seed["initial_liquidity"] * (1 + np.random.normal(scale=0.3)))
    init_marketcap = max(1000.0, seed["initial_marketcap"] * (1 + np.random.normal(scale=0.3)))
    base_holders = max(1, int(seed["base_holders"] * (1 + np.random.normal(scale=0.2))))
    base_conc = min(max(0.01, seed["base_conc"] * (1 + np.random.normal(scale=0.2))), 0.99)

    # price path: allow early drift then cooling
    mu_profile = [mu_early if t < 10 else mu_early * (0.4 if label=="good" else 0.6) for t in range(minutes)]
    sigma_profile = [sigma_early if t < 10 else sigma_early * 0.7 for t in range(minutes)]
    prices = [init_price]
    for t in range(1, minutes+1):
        mu = mu_profile[min(t-1, len(mu_profile)-1)]
        sigma = sigma_profile[min(t-1, len(sigma_profile)-1)]
        z = np.random.normal()
        S_prev = prices[-1]
        S_new = S_prev * math.exp((mu - 0.5 * sigma**2) + sigma * z)
        prices.append(max(S_new, 1e-12))
    # prices is length 61 (0..60)
    # Build per-minute volume: correlate with price change magnitude and liquidity
    price_changes = np.abs(np.diff(prices))
    # base volume proportional to liquidity and volatility; good tokens have higher volume early
    vol_base = init_liquidity * 0.05
    volumes = []
    for t in range(minutes):
        vol = vol_base * (1 + (price_changes[t] / (prices[t] + 1e-12)) * 50)
        # extra early spike for good patterns
        if label == "good" and t < 10:
            vol *= random.uniform(1.5, 4.0)
        # noise
        vol = vol * random.uniform(0.6, 1.6)
        volumes.append(max(vol, 0.0))
    # holders trajectory
    holders = sample_holder_counts(base_holders, minutes, holder_growth)
    # concentration trajectory
    conc = wallet_concentration_top10(base_conc, minutes, conc_decay)
    # buys/sells per minute
    buys, sells = simulate_transaction_counts(volumes, aggressiveness if label=="good" else 0.6)
    # liquidity timeline: simplistic: initial liquidity grows with buys early, then stabilizes
    liquidity = []
    liq = init_liquidity
    for t in range(minutes+1):
        # convert buys[t-1] influence to liquidity increase
        if t == 0:
            liquidity.append(liq)
            continue
        # delta proportional to buys-sells
        delta = (buys[t-1] - sells[t-1]) * prices[t] * 0.5
        # for good, initial liquidity injection is larger if t == liquidity_added_at_minute
        if t-1 == liquidity_added_at_minute:
            delta += init_liquidity * random.uniform(0.2, 1.5)
        liq = max(0.0, liq + delta * random.uniform(0.3, 1.0))
        liquidity.append(liq)

    # Aggregations for 10m/30m/60m windows (we'll use minute indices)
    # progress_pct_Xm: percent change from minute 0 open to X-minute close
    def pct_change_at(minute):
        open_price = prices[0]
        close_price = prices[minute]
        return float((close_price / open_price - 1.0) * 100.0)

    progress_10 = pct_change_at(10)
    progress_30 = pct_change_at(30)
    progress_60 = pct_change_at(60)
    lp_10m = float(liquidity[10])
    lp_30m = float(liquidity[30])
    lp_60m = float(liquidity[60])
    market_cap_10m = float(lp_10m * prices[10]) if lp_10m and prices[10] else float(init_marketcap * (1 + progress_10/100.0))
    market_cap_30m = float(lp_30m * prices[30]) if lp_30m and prices[30] else float(init_marketcap * (1 + progress_30/100.0))
    market_cap_60m = float(lp_60m * prices[60]) if lp_60m and prices[60] else float(init_marketcap * (1 + progress_60/100.0))

    holders_10 = int(holders[10])
    holders_30 = int(holders[30])
    holders_60 = int(holders[60])
    conc_10 = float(conc[10])
    conc_30 = float(conc[30])
    conc_60 = float(conc[60])

    # buy/sell counts aggregation:
    buy_count_5m_30m = sum(buys[0:5])   # buys in first 5 minutes observed at 30m
    buy_count_60m = sum(buys[0:60])
    sell_count_5m_30m = sum(sells[0:5])
    sell_count_60m = sum(sells[0:60])
    volume_5m_30m = float(sum(volumes[0:5]))
    volume_60m = float(sum(volumes[0:60]))

    # website & socials heuristics:
    # good tokens more likely to have socials; bad tokens less likely. Add randomness.
    has_website = 1 if (label=="good" and random.random() < 0.85) or (label=="bad" and random.random() < 0.2) else 0
    has_twitter = 1 if (label=="good" and random.random() < 0.7) or (label=="bad" and random.random() < 0.25) else 0
    has_telegram = 1 if (label=="good" and random.random() < 0.75) or (label=="bad" and random.random() < 0.3) else 0

    # OHLCV for 30m and 60m windows: compute from minute-level series for the window starting at 0
    def ohlcv_for_window(prices_list, volumes_list, window_end_min):
        # open = price at minute 0, close = price at minute window_end_min
        open_p = float(prices_list[0])
        close_p = float(prices_list[window_end_min])
        high_p = float(max(prices_list[0:window_end_min+1]))
        low_p = float(min(prices_list[0:window_end_min+1]))
        vol = float(sum(volumes_list[0:window_end_min]))
        return open_p, high_p, low_p, close_p, vol

    open_30, high_30, low_30, close_30, vol_30 = ohlcv_for_window(prices, volumes, 30)
    open_60, high_60, low_60, close_60, vol_60 = ohlcv_for_window(prices, volumes, 60)

    pattern = {
        "liquidity_added_at_minute": int(liquidity_added_at_minute),
        "lp_10m": float(lp_10m),
        "lp_30m": float(lp_30m),
        "lp_60m": float(lp_60m),
        "market_cap_10m": float(market_cap_10m),
        "market_cap_30m": float(market_cap_30m),
        "market_cap_60m": float(market_cap_60m),
        "progress_pct_10m": float(progress_10),
        "progress_pct_30m": float(progress_30),
        "progress_pct_60m": float(progress_60),
        "holders_count_10m": holders_10,
        "holders_count_30m": holders_30,
        "holders_count_60m": holders_60,
        "wallet_concentration_top10_10m": float(conc_10),
        "wallet_concentration_top10_30m": float(conc_30),
        "wallet_concentration_top10_60m": float(conc_60),
        "buy_count_5m_30m": int(buy_count_5m_30m),
        "buy_count_60m": int(buy_count_60m),
        "sell_count_5m_30m": int(sell_count_5m_30m),
        "sell_count_60m": int(sell_count_60m),
        "volume_5m_30m": float(volume_5m_30m),
        "volume_60m": float(volume_60m),
        "has_website_10m": has_website,
        "has_website_30m": has_website,
        "has_website_60m": has_website,
        "has_twitter_10m": has_twitter,
        "has_twitter_30m": has_twitter,
        "has_twitter_60m": has_twitter,
        "has_telegram_10m": has_telegram,
        "has_telegram_30m": has_telegram,
        "has_telegram_60m": has_telegram,
        "open_30m": float(open_30),
        "high_30m": float(high_30),
        "low_30m": float(low_30),
        "close_30m": float(close_30),
        "ohlcv_volume_30m": float(vol_30),
        "open_60m": float(open_60),
        "high_60m": float(high_60),
        "low_60m": float(low_60),
        "close_60m": float(close_60),
        "ohlcv_volume_60m": float(vol_60),
        "label": label
    }

    return pattern

# -----------------------
# GENERATE MANY PATTERNS
# -----------------------
def generate_dataset(n_good: int, n_bad: int) -> List[Dict]:
    out = []
    # generate good
    for i in range(n_good):
        seed = random.choice(GOOD_SEEDS)
        pat = generate_single_pattern(seed, label="good")
        out.append(pat)
    # generate bad
    for i in range(n_bad):
        seed = random.choice(BAD_SEEDS)
        pat = generate_single_pattern(seed, label="bad")
        out.append(pat)
    random.shuffle(out)
    return out

# -----------------------
# ENGINEER DELTAS & RATIOS (match core/analyzer expectations)
# -----------------------
def add_deltas_and_ratios(patterns: List[Dict]) -> List[Dict]:
    windows = [("10m","30m"), ("30m","60m")]  # we generate features for 10->30 and 30->60
    metrics = ["lp", "market_cap", "progress_pct", "holders_count", "wallet_concentration_top10"]
    for pat in patterns:
        for w0, w1 in windows:
            for m in metrics:
                f0 = f"{m}_{w0}"
                f1 = f"{m}_{w1}"
                dname = f"{m}_delta_{w1}_{w0}"
                rname = f"{m}_ratio_{w1}_{w0}"
                v0 = pat.get(f0, 0.0)
                v1 = pat.get(f1, 0.0)
                pat[dname] = float(v1 - v0)
                # safe ratio: if v0 is zero, set ratio to v1 (or 0)
                if v0 == 0:
                    pat[rname] = float(v1) if v1 != 0 else 0.0
                else:
                    pat[rname] = float(v1 / v0)
    return patterns


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating {N_GOOD} good and {N_BAD} bad synthetic patterns (this may take ~1-3 minutes)...")
    dataset = generate_dataset(N_GOOD, N_BAD)
    dataset = add_deltas_and_ratios(dataset)
    # Save as JSON list
    with open(OUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} patterns to {OUT_FILE}")

if __name__ == "__main__":
    main()