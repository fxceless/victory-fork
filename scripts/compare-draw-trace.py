#!/usr/bin/env python3
"""Compare the stable binding roles in two VICTORY_TRACE_DRAW captures."""

import argparse
import bisect
import re
from pathlib import Path

MAX_FENCE = (1 << 64) - 1
FIELDS = set("role title_slot host_slot submit_slot title record_state record_object "
             "record_desc submit_state submit_object submit_desc upload_fence upload_order upload_us upload_sample".split())


def read_trace(path):
    meta, bindings, header = {}, {}, None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split("\t")
        if header is None:
            if fields[0] == "role":
                if not FIELDS.issubset(fields):
                    raise ValueError(f"{path}: incomplete bindings header")
                header = fields
            elif len(fields) == 2 and fields[0] != "detail":
                meta[fields[0]] = fields[1]
            continue
        if len(fields) != len(header) or fields[0] in bindings:
            raise ValueError(f"{path}: malformed or duplicate binding {fields[0]}")
        bindings[fields[0]] = dict(zip(header, fields))
    if meta.get("version") != "1" or header is None or not meta.get("signature"):
        raise ValueError(f"{path}: expected a version 1 draw trace")

    completed, times = [], []
    timeline = path.parent / Path(meta["fence_file"]).name
    if timeline.exists():
        for line in timeline.read_text(encoding="utf-8-sig").splitlines()[1:]:
            value, time = map(int, line.split("\t"))
            if value == MAX_FENCE or (completed and (value <= completed[-1] or time < times[-1])):
                raise ValueError(f"{timeline}: invalid completion timeline")
            completed.append(value)
            times.append(time)
    later = {}
    next_path = path.parent / Path(meta["follow_file"]).name
    if next_path.exists():
        for line in next_path.read_text(encoding="utf-8-sig").splitlines():
            token, fence, order, when, sample = line.split("\t")
            later[token] = (int(fence), int(order), int(when), sample)
    return meta, bindings, (completed, times, later)


def readiness(row, meta):
    if row["upload_fence"] == "-":
        return "no tracked staging copy"
    upload, draw = int(row["upload_fence"]), int(meta["draw_fence"])
    done = int(meta["completed_before"])
    if upload == 0 or done == MAX_FENCE or meta["signal_ok"] != "true":
        return "fence unavailable"
    if upload == draw and int(row["upload_order"]) < int(meta["event_order"]):
        return "copy precedes draw in the same submission"
    if upload < draw:
        return "completed before draw submission" if done >= upload else "earlier submission in flight at enqueue"
    return "inconsistent queue order"


def completion(row, meta, timeline):
    state = readiness(row, meta)
    if state in ("no tracked staging copy", "fence unavailable", "inconsistent queue order"):
        return state
    values, times, _ = timeline
    pos = bisect.bisect_left(values, int(row["upload_fence"]))
    bounds = times[pos:pos + 1]
    if state == "completed before draw submission":
        bounds = bounds + [int(meta["execute_us"])]
    observed = f", complete by {min(bounds)} us" if bounds else ", completion unobserved"
    return f"{state}, copy fence {row['upload_fence']} at {row['upload_us']} us{observed}"


def texture(row):
    return row and ("title-texture" in row["title"]
                    or row["submit_desc"].startswith(("d2:", "d3:", "d4:")))


def changes(role, good, bad, gm, bm):
    if good is None or bad is None:
        return ["role missing in one capture"]
    reasons = []
    if (good["submit_state"], good["submit_desc"]) != (bad["submit_state"], bad["submit_desc"]):
        reasons.append("host resource state or description changed")
    if (good["record_state"], good["record_desc"]) != (bad["record_state"], bad["record_desc"]):
        reasons.append("recorded resource state or description changed")
    if rebound(good) != rebound(bad):
        reasons.append("binding changed between recording and submission in one run")
    for key in ("view_format", "view_levels"):
        a, b = (re.search(rf"\b{key}=(\S+)", row["title"]) for row in (good, bad))
        if a and b and a.group(1) != b.group(1):
            reasons.append(f"title {key} changed")
    a, b = good["upload_sample"], bad["upload_sample"]
    if a != "-" and b != "-" and a != b:
        reasons.append("sampled upload texels changed")
    if readiness(good, gm) != readiness(bad, bm):
        reasons.append("staging-copy evidence or fence state changed")
    if role.startswith("word/") and good["submit_slot"] != bad["submit_slot"]:
        reasons.append("material word selected a different heap index")
    return reasons


def rebound(row):
    return (row["record_object"], row["record_state"]) != (row["submit_object"], row["submit_state"])


def describe(row, meta, timeline):
    if row is None:
        return "role absent"
    state = row["submit_state"]
    annotations = []
    if rebound(row):
        annotations.append("host object/state changed since recording")
    if row["host_slot"] != row["submit_slot"]:
        annotations.append(f"recorded heap index {row['host_slot']}")
    extra = f" ({', '.join(annotations)})" if annotations else ""
    values, times, later = timeline
    future = ""
    if row.get("watch") in later:
        fence, order, when, sample = later[row["watch"]]
        pos = bisect.bisect_left(values, fence)
        upper = f", complete by {times[pos]} us" if fence and pos < len(times) else ", completion unobserved"
        future = f", next staging copy fence {fence} event {order} at {when} us, sample {sample}{upper}"
    return (f"heap {row['submit_slot']}, {state} {row['submit_desc']}{extra}, "
            f"upload sample {row['upload_sample']}, {completion(row, meta, timeline)}{future}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("good", type=Path)
    parser.add_argument("bad", type=Path)
    parser.add_argument("--role", help="show an exact role even if its observations match")
    args = parser.parse_args()
    try:
        gm, good, gt = read_trace(args.good)
        bm, bad, bt = read_trace(args.bad)
        if gm["signature"] != bm["signature"]:
            raise ValueError("draw signatures differ; identify the same colour draw in both captures")
        roles = good.keys() | bad.keys()
        if args.role and args.role not in roles:
            raise ValueError(f"unknown role {args.role}")
        print(gm["signature"])
        for label, meta in (("good", gm), ("bad", bm)):
            print(f"  {label}: frame {meta['frame']}, occurrence {meta['occurrence']}, "
                  f"draw fence {meta['draw_fence']}, execute {meta['execute_us']} us, "
                  f"completed before enqueue {meta['completed_before']}")
        if not roles:
            raise ValueError("neither capture contains descriptor roles")
        reported = 0
        for role in sorted([args.role] if args.role else roles):
            a, b = good.get(role), bad.get(role)
            if role.startswith("word/") and not (texture(a) or texture(b) or args.role):
                continue
            reasons = changes(role, a, b, gm, bm)
            if a and b:
                later = [timeline[2].get(row.get("watch")) for row, timeline in ((a, gt), (b, bt))]
                if ((later[0] is None) != (later[1] is None)
                        or (all(later) and later[0][3] != later[1][3])):
                    reasons.append("later staging-copy observations differ")
            if not reasons and not args.role:
                continue
            kind = "candidate material word" if role.startswith("word/") else "descriptor binding"
            print(f"\n{role} [{kind}]: {', '.join(reasons) or 'observations match'}")
            print("  good:", describe(a, gm, gt))
            print("  bad: ", describe(b, bm, bt))
            reported += 1
        if not reported:
            print("No difference observed in the comparable bindings and upload samples.")
    except (OSError, ValueError, KeyError) as err:
        parser.exit(2, f"Draw comparison failed: {err}\n")


if __name__ == "__main__":
    main()
