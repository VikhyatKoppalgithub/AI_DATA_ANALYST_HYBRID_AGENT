"""Fetch NYC Citywide Payroll — the real file that found five bugs.

    python scripts/fetch_nyc_payroll.py

Both bundled fixtures are synthetic, so they contain only the mess I thought to
add. This one is a genuine government export, and running it through the
pipeline surfaced five failures the fixtures could not (see the README).

Downloaded rather than committed: FY2024 + FY2025 is 1,113,117 rows and 211 MB,
which does not belong in a git repository. The endpoint is public and needs no
key. Like the fixture generators, this script prints the ground truth it expects
so the findings can be checked rather than taken on faith.

Source: https://data.cityofnewyork.us/City-Government/Citywide-Payroll-Data-Fiscal-Year-/k397-673e
Licence: NYC Open Data, public domain.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "nyc_payroll_fy2024_2025.csv"
RESOURCE = "https://data.cityofnewyork.us/resource/k397-673e.csv"
YEARS = ("2024", "2025")
# One request for both years. The rows arrive grouped by agency, so a smaller
# $limit does not give a usable sample — 5,000 rows returned a single agency and
# a single fiscal year. It is the whole two years or nothing.
LIMIT = 1_200_000
EXPECTED_ROWS = 1_113_117


def build_url() -> str:
    where = urllib.parse.quote(f"fiscal_year in ('{YEARS[0]}','{YEARS[1]}')")
    return f"{RESOURCE}?$where={where}&$limit={LIMIT}"


def fetch(url: str, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with urllib.request.urlopen(url, timeout=1800) as response, destination.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
            written += len(chunk)
            print(f"\r  {written / 1e6:7.1f} MB", end="", flush=True)
    print()
    return written


def main() -> int:
    url = build_url()
    print(f"Fetching FY{YEARS[0]}-FY{YEARS[1]} payroll -> {OUT}")
    print("(~211 MB; the whole two years, because the rows arrive grouped by agency)")
    try:
        fetch(url, OUT)
    except urllib.error.URLError as exc:
        print(f"\nCould not reach NYC Open Data: {exc.reason}", file=sys.stderr)
        return 1

    rows = sum(1 for _ in OUT.open()) - 1
    print(f"\n{rows:,} rows written")
    if rows != EXPECTED_ROWS:
        print(
            f"NOTE: expected {EXPECTED_ROWS:,}. The city revises this export, so a "
            "drift of a few rows is normal; the figures below may shift slightly."
        )

    print(
        """
What this file is for — the five failures it found, and what to look for:

  1. metric naming     The decoy warning hardcoded "revenue" and told a payroll
                       analysis a segment was "1.1% of revenue".

  2. offsetting        Ask "how did total overtime pay change in fiscal year
                       2025?". Overtime falls -4.47% net, but that is
                       -8.75pp of falls against +4.27pp of rises:
                         POLICE DEPARTMENT        -5.63 pp
                         DEPARTMENT OF CORRECTION +2.64 pp
                         DEPARTMENT OF SANITATION +1.29 pp
                       The concentration check reported the top contributor as
                       explaining 126% of the move -- and passed, because it is
                       one-sided. Above 100% is offsetting, not concentration.

  3. cardinality       `last_name` holds 125,431 distinct values (11% of rows)
                       and profiles as categorical, so it was accepted as a
                       dimension. Grouping by it yields ~600,000 segments.

  4. coded IDs         `payroll_number` is 159 agency codes. It escapes the
                       near-uniqueness identifier rule, so it was the first
                       numeric column a hallucinated metric fell back to.

  5. competing axes    `agency_start_date` holds hire dates back to 1901 and
                       sorts ahead of `fiscal_year`, so the planner was shown
                       1901-2025 as the range for a file covering FY2024-25.

Reproduce:
  python -m analyst.cli data/nyc_payroll_fy2024_2025.csv \\
      "How did total overtime pay change in fiscal year 2025?"
"""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
