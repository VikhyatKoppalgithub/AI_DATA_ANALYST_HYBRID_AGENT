"""Generate a second fixture: customer-support operations.

Deliberately unlike data/sales_2026.csv along every axis that matters, so the
pipeline is not being validated against one shape of problem:

  domain      support operations, not retail sales
  metric      a count of SLA breaches, not currency
  dates       ISO 8601 with a timezone, not MM/DD/YYYY
  nullability a genuinely nullable datetime (open tickets have no closed_at)
  types       text booleans, an ordinal priority, a duration with nulls
  mess        different null tokens (NULL, n/a, TBD), a numeric-looking ID

The March shock keeps the same two properties as the sales fixture, because
they are what the evals test: one large real contributor, and one decoy whose
percentage move is dramatic and whose contribution is negligible.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 4417
OUT = Path(__file__).resolve().parents[1] / "data" / "support_tickets_2026.csv"

TEAMS = ["Platform", "Billing", "Onboarding", "Integrations"]
PRIORITIES = ["P1", "P2", "P3", "P4"]
TIERS = ["Enterprise", "Business", "Starter"]
CHANNELS = ["email", "chat", "phone"]

# Monthly ticket volume per (team, priority). Kept large on purpose: breach
# counts are Poisson-ish, so a slice with five breaches swings +/-60% on noise
# alone and the "ground truth" stops being ground truth.
VOLUME = {
    "Platform": {"P1": 450, "P2": 750, "P3": 1050, "P4": 600},
    "Billing": {"P1": 100, "P2": 350, "P3": 650, "P4": 475},
    "Onboarding": {"P1": 100, "P2": 300, "P3": 550, "P4": 400},
    "Integrations": {"P1": 175, "P2": 425, "P3": 600, "P4": 350},
}

# Baseline probability a ticket breaches its SLA.
BREACH_RATE = {"P1": 0.13, "P2": 0.09, "P3": 0.05, "P4": 0.03}

# Onboarding runs an unusually tight P1 queue. This is what makes it a usable
# decoy: a low base rate on a small volume means ~3 breaches a month, so a
# sixfold jump is a huge percentage and a trivial share of the total.
BREACH_RATE_OVERRIDE = {("Onboarding", "P1"): 0.03}

# Target resolution hours by priority (lognormal-ish spread around these).
RESOLUTION_HOURS = {"P1": 5.0, "P2": 14.0, "P3": 38.0, "P4": 72.0}

# March: Platform P1 is the real story — a big slice with a big jump.
# Billing P4 is the decoy: rate quadruples, but on a small base of low-priority
# tickets, so it moves the total barely at all.
MARCH_BREACH_MULTIPLIER = {
    ("Platform", "P1"): 4.3,
    ("Onboarding", "P1"): 6.0,   # decoy: huge percentage, negligible contribution
    ("Integrations", "P2"): 1.12,
}

MONTHS = [(2026, m) for m in range(1, 6)]
NULL_TOKENS = ["NULL", "n/a", "TBD", ""]


def build(rng: random.Random, nprng: np.random.Generator) -> pd.DataFrame:
    rows = []
    ticket = 500_000

    for year, month in MONTHS:
        days = pd.Period(f"{year}-{month:02d}").days_in_month
        for team in TEAMS:
            for priority in PRIORITIES:
                volume = VOLUME[team][priority]
                count = max(1, int(nprng.normal(volume, volume * 0.02)))
                rate = BREACH_RATE_OVERRIDE.get((team, priority), BREACH_RATE[priority])
                if month == 3:
                    rate *= MARCH_BREACH_MULTIPLIER.get((team, priority), 1.0)
                rate = min(rate, 0.97)

                for _ in range(count):
                    ticket += 1
                    day = rng.randint(1, days)
                    hour, minute = rng.randint(0, 23), rng.randint(0, 59)
                    opened = pd.Timestamp(
                        year=year, month=month, day=day, hour=hour, minute=minute, tz="UTC"
                    )

                    breached = nprng.random() < rate
                    target = RESOLUTION_HOURS[priority]
                    # Breached tickets take materially longer to resolve.
                    scale = 1.9 if breached else 0.75
                    hours = float(nprng.lognormal(np.log(target * scale), 0.42))

                    # ~6% of tickets are still open: no closed_at, no resolution.
                    still_open = nprng.random() < 0.06
                    closed = None if still_open else opened + pd.Timedelta(hours=hours)

                    rows.append(
                        {
                            "ticket_id": ticket,  # numeric-looking, must not be summed
                            "opened_at": opened.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "closed_at": (
                                closed.strftime("%Y-%m-%dT%H:%M:%SZ") if closed is not None else None
                            ),
                            "team": team,
                            "priority": priority,
                            "customer_tier": rng.choices(TIERS, weights=[0.3, 0.45, 0.25])[0],
                            "channel": rng.choices(CHANNELS, weights=[0.5, 0.35, 0.15])[0],
                            "resolution_hours": None if still_open else round(hours, 2),
                            "sla_breach": 0 if still_open else int(breached),
                            "reopened": "Yes" if (not still_open and nprng.random() < 0.08) else "No",
                            "satisfaction": (
                                None
                                if still_open or nprng.random() < 0.35
                                else int(np.clip(nprng.normal(3.4 if breached else 4.3, 0.9), 1, 5))
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def messify(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    df = df.copy()

    # Inconsistent casing and padding on a dimension the agent must group by.
    idx = df.sample(frac=0.07, random_state=3).index
    df.loc[idx, "team"] = df.loc[idx, "team"].str.upper()
    idx = df.sample(frac=0.03, random_state=5).index
    df.loc[idx, "team"] = "  " + df.loc[idx, "team"] + " "

    # Null-ish text tokens rather than real NaN — different set to the sales file.
    idx = df.sample(frac=0.025, random_state=7).index
    df.loc[idx, "customer_tier"] = rng.choices(NULL_TOKENS, k=len(idx))
    df["closed_at"] = df["closed_at"].fillna("NULL")

    # Duplicated rows, as from a re-run export.
    dupes = df.sample(frac=0.015, random_state=11)
    df = pd.concat([df, dupes], ignore_index=True)

    return df.sample(frac=1.0, random_state=13).reset_index(drop=True)


def report_ground_truth(clean: pd.DataFrame) -> None:
    df = clean.copy()
    df["month"] = pd.to_datetime(df["opened_at"], format="%Y-%m-%dT%H:%M:%SZ").dt.month
    feb, mar = df[df.month == 2], df[df.month == 3]
    before, after = feb["sla_breach"].sum(), mar["sla_breach"].sum()
    delta = after - before

    print(f"\n{'=' * 66}\nGROUND TRUTH — SLA breaches, Feb to Mar 2026\n{'=' * 66}")
    print(f"February breaches : {before:>6}")
    print(f"March breaches    : {after:>6}")
    print(f"Change            : {delta:>+6}  ({100 * delta / before:+.1f}%)")

    f = feb.groupby(["team", "priority"])["sla_breach"].sum()
    m = mar.groupby(["team", "priority"])["sla_breach"].sum()
    contrib = ((m - f) / before * 100).sort_values(ascending=False)
    print("\nTop contributing team x priority slices (percentage points):")
    for (team, priority), pp in contrib.head(5).items():
        own = 100 * (m[(team, priority)] / f[(team, priority)] - 1) if f[(team, priority)] else float("nan")
        print(f"  {team:<14} {priority:<4} {pp:>7.2f} pp   (slice itself {own:+.1f}%, "
              f"{f[(team, priority)]} -> {m[(team, priority)]})")

    print("\nOther facts (for the code-route eval cases):")
    print(f"  median resolution_hours       : {clean['resolution_hours'].median():.2f}")
    print(f"  mean resolution_hours         : {clean['resolution_hours'].mean():.2f}")
    print(f"  still-open tickets (no close) : {int(clean['closed_at'].isna().sum())}")
    print(f"  reopened == Yes               : {int((clean['reopened'] == 'Yes').sum())}")
    print(f"  distinct teams                : {clean['team'].nunique()}")
    corr = clean[["resolution_hours", "satisfaction"]].dropna().corr().iloc[0, 1]
    print(f"  corr(resolution_hours, satis) : {corr:.4f}")
    print(f"{'=' * 66}\n")


def main() -> None:
    rng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)

    clean = build(rng, nprng)
    messy = messify(clean, rng)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    messy.to_csv(OUT, index=False)
    print(f"Wrote {len(messy):,} rows to {OUT}")
    report_ground_truth(clean)


if __name__ == "__main__":
    main()
