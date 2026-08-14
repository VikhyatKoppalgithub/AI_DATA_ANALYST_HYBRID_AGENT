"""Generate a messy sales dataset with a known-correct answer.

March total revenue is engineered to fall ~14% against February, with the
decline concentrated in West-region laptops. Because the ground truth is known,
this file doubles as the first eval case: the agent's contribution analysis can
be scored against the numbers printed at the end of this script.

The mess is deliberate and mirrors real exports: dates and currency stored as
text, inconsistent casing, null-ish tokens, duplicate rows, a trailing unnamed
column, and an ID column that must not be aggregated.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260313
OUT = Path(__file__).resolve().parents[1] / "data" / "sales_2026.csv"

REGIONS = ["West", "East", "Central", "South"]
PRODUCTS = ["Laptop", "Monitor", "Keyboard", "Docking Station"]
CHANNELS = ["Retail", "Direct"]

UNIT_PRICE = {"Laptop": 1450.0, "Monitor": 320.0, "Keyboard": 85.0, "Docking Station": 210.0}

# Baseline order volume per (region, product) per month.
BASE_VOLUME = {
    "West": {"Laptop": 220, "Monitor": 140, "Keyboard": 190, "Docking Station": 110},
    "East": {"Laptop": 150, "Monitor": 120, "Keyboard": 160, "Docking Station": 90},
    "Central": {"Laptop": 95, "Monitor": 85, "Keyboard": 120, "Docking Station": 70},
    "South": {"Laptop": 80, "Monitor": 70, "Keyboard": 100, "Docking Station": 55},
}

# March shock. Two things are being set up here:
#
#   1. West laptops are the real story — a large slice taking a large hit, so it
#      dominates the contribution ranking.
#   2. South docking stations are a decoy — the largest *percentage* drop in the
#      dataset, in a slice too small to matter. An agent that leads with it has
#      confused "biggest percentage change" with "biggest contributor", which is
#      the failure this fixture exists to catch.
MARCH_MULTIPLIER = {
    ("West", "Laptop"): 0.62,
    ("South", "Docking Station"): 0.45,  # decoy
    ("East", "Laptop"): 0.97,
    ("Central", "Monitor"): 0.96,
}

MONTHS = [(2026, m) for m in range(1, 6)]
MONTH_TREND = {1: 1.00, 2: 1.03, 3: 1.02, 4: 1.05, 5: 1.07}  # mild growth, pre-shock


def build_orders(rng: random.Random, nprng: np.random.Generator) -> pd.DataFrame:
    rows = []
    order_id = 100_000

    for year, month in MONTHS:
        days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
        for region in REGIONS:
            for product in PRODUCTS:
                volume = BASE_VOLUME[region][product] * MONTH_TREND[month]
                if month == 3:
                    volume *= MARCH_MULTIPLIER.get((region, product), 1.0)
                # Sampling noise has to stay well below the injected shocks, or
                # the ground truth this file exists to provide is not reliable.
                n_orders = max(1, int(nprng.normal(volume, volume * 0.015)))

                for _ in range(n_orders):
                    order_id += 1
                    day = rng.randint(1, days_in_month)
                    # gamma(6, 0.45): same ~2.7 mean as a wider shape, but a
                    # coefficient of variation of 0.41 instead of 0.71, so the
                    # month-to-month drift in mean units per order stays small.
                    units = max(1, round(nprng.gamma(6.0, 0.45)))
                    # Small per-order price variation from discounting.
                    price = UNIT_PRICE[product] * nprng.uniform(0.96, 1.02)
                    rows.append(
                        {
                            "order_id": f"SO-{order_id}",
                            "order_date": f"{month:02d}/{day:02d}/{year}",
                            "region": region,
                            "product": product,
                            "channel": rng.choices(CHANNELS, weights=[0.62, 0.38])[0],
                            "units": units,
                            "unit_price": round(price, 2),
                            "revenue": round(units * price, 2),
                            "sales_rep": f"rep_{rng.randint(1, 40):02d}",
                        }
                    )

    return pd.DataFrame(rows)


def messify(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    df = df.copy()

    # Currency as text, with thousands separators.
    df["revenue"] = df["revenue"].map(lambda v: f"${v:,.2f}")
    df["unit_price"] = df["unit_price"].map(lambda v: f"${v:,.2f}")

    # Inconsistent casing and whitespace in a dimension the agent must group by.
    idx = df.sample(frac=0.08, random_state=7).index
    df.loc[idx, "region"] = df.loc[idx, "region"].str.lower()
    idx = df.sample(frac=0.04, random_state=11).index
    df.loc[idx, "region"] = " " + df.loc[idx, "region"] + " "

    # Null-ish tokens rather than real NaN.
    idx = df.sample(frac=0.02, random_state=13).index
    df.loc[idx, "channel"] = rng.choices(["N/A", "", "unknown", "-"], k=len(idx))

    # Genuinely missing sales rep on some rows.
    idx = df.sample(frac=0.03, random_state=17).index
    df.loc[idx, "sales_rep"] = np.nan

    # A trailing all-null column, the classic spreadsheet artifact.
    df["Unnamed: 9"] = np.nan

    # Duplicate rows, as happens when an export is run twice and concatenated.
    dupes = df.sample(frac=0.02, random_state=23)
    df = pd.concat([df, dupes], ignore_index=True)

    return df.sample(frac=1.0, random_state=29).reset_index(drop=True)


def report_ground_truth(clean: pd.DataFrame) -> None:
    """Print the answer the agent is supposed to find."""
    df = clean.copy()
    df["month"] = pd.to_datetime(df["order_date"], format="%m/%d/%Y").dt.month

    feb, mar = df[df.month == 2], df[df.month == 3]
    feb_total, mar_total = feb["revenue"].sum(), mar["revenue"].sum()
    delta = mar_total - feb_total
    pct = 100 * delta / feb_total

    print(f"\n{'=' * 62}\nGROUND TRUTH — Feb to Mar 2026\n{'=' * 62}")
    print(f"February revenue : ${feb_total:>14,.2f}")
    print(f"March revenue    : ${mar_total:>14,.2f}")
    print(f"Change           : ${delta:>14,.2f}  ({pct:+.1f}%)")

    for dim in ("region", "product"):
        f = feb.groupby(dim)["revenue"].sum()
        m = mar.groupby(dim)["revenue"].sum()
        contrib = ((m - f) / feb_total * 100).sort_values()
        print(f"\nContribution to the total change, by {dim} (percentage points):")
        for name, pp in contrib.items():
            print(f"  {name:<18} {pp:>7.2f} pp")

    f = feb.groupby(["region", "product"])["revenue"].sum()
    m = mar.groupby(["region", "product"])["revenue"].sum()
    contrib = ((m - f) / feb_total * 100).sort_values()
    print("\nTop 5 contributing region x product slices:")
    for (region, product), pp in contrib.head(5).items():
        own = 100 * (m[(region, product)] - f[(region, product)]) / f[(region, product)]
        print(f"  {region:<9} {product:<16} {pp:>7.2f} pp   (slice itself {own:+.1f}%)")
    print(f"{'=' * 62}\n")


def main() -> None:
    rng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)

    clean = build_orders(rng, nprng)
    messy = messify(clean, rng)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    messy.to_csv(OUT, index=False)

    print(f"Wrote {len(messy):,} rows to {OUT}")
    report_ground_truth(clean)


if __name__ == "__main__":
    main()
