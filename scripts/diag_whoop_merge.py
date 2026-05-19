"""Read-only diagnostic: dump raw WHOOP v2 records + replay merge linking.

Throwaway. Prints recovery/cycle/sleep ids + types + score_state for the
last N days and shows, per recovery, whether the cycle/sleep join lands.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

from src.ingest.normalizer import _local_date, merge_history
from src.ingest.whoop_client import WhoopClient

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 7


def t(v):
    return f"{v!r}<{type(v).__name__}>"


def main() -> None:
    end = date.today()
    start = end - timedelta(days=DAYS - 1)
    with WhoopClient() as c:
        recs = c.get_recovery(start, end)
        cycs = c.get_cycles(start, end)
        slps = c.get_sleep(start, end)

    print(f"\n=== window {start}..{end}  "
          f"recoveries={len(recs)} cycles={len(cycs)} sleeps={len(slps)} ===\n")

    print("-- CYCLES --")
    for x in sorted(cycs, key=lambda r: r.get("start") or ""):
        off = x.get("timezone_offset")
        ld = _local_date(x["start"], off) if x.get("start") else "?"
        print(f"  id={t(x.get('id'))} start={x.get('start')} "
              f"tz={off!r} -> local_date={ld} "
              f"end={x.get('end')} state={x.get('score_state')}")

    print("\n-- SLEEPS --")
    for x in sorted(slps, key=lambda r: r.get("start") or ""):
        print(f"  id={t(x.get('id'))} start={x.get('start')} "
              f"nap={x.get('nap')} state={x.get('score_state')}")

    print("\n-- RECOVERIES (join replay) --")
    cycle_by_id = {c.get("id"): c for c in cycs if c.get("id") is not None}
    sleep_by_id = {s.get("id"): s for s in slps if s.get("id") is not None}
    for r in sorted(recs, key=lambda r: r.get("created_at") or ""):
        cid, sid = r.get("cycle_id"), r.get("sleep_id")
        cyc = cycle_by_id.get(cid)
        slp = sleep_by_id.get(sid)
        print(f"  created={r.get('created_at')} state={r.get('score_state')}")
        print(f"    cycle_id={t(cid)} -> "
              f"{'HIT ' + (cyc.get('start') or '?') if cyc else 'MISS'}"
              f" (state={cyc.get('score_state') if cyc else '-'})")
        print(f"    sleep_id={t(sid)} -> "
              f"{'HIT ' + (slp.get('start') or '?') if slp else 'MISS'}")
        if cyc and cyc.get("start"):
            print(f"    => by_date key (local_date of cycle) = "
                  f"{_local_date(cyc['start'], cyc.get('timezone_offset'))}")

    print("\n-- merge_history() ACTUAL OUTPUT --")
    series = merge_history(start, end, WhoopClient())
    print(f"  {len(series)} day(s): "
          + ", ".join(f"{w.date}(strain={w.day_strain})" for w in series))


if __name__ == "__main__":
    main()
