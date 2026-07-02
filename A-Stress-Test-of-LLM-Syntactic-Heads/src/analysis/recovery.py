"""H2: does confidence bottom out right at the disambiguation token, and does
it recover afterwards or stay depressed ("structural poisoning")?

We look at a window of +/- 4 tokens around the disambiguation point. offset=0
is the disambiguating token itself, offset=-1 is pre-disambiguation, +1 is
post-disambiguation.
"""
import numpy as np

WINDOW = 4


def recovery_analysis(records):
    """records: list of dicts with keys 'offset' (int, relative to
    disambiguation token) and 'conf_max' (float).
    Returns mean confidence at each offset in [-WINDOW, WINDOW].
    """
    conf_by_offset = {o: [] for o in range(-WINDOW, WINDOW + 1)}
    for r in records:
        o = r.get("offset")
        if o is not None and -WINDOW <= o <= WINDOW and not np.isnan(r["conf_max"]):
            conf_by_offset[int(o)].append(r["conf_max"])
    return {o: float(np.mean(v)) for o, v in conf_by_offset.items() if v}


def crash_and_recovery(conf_by_offset):
    """Pre = offset -1, At = offset 0, Post = offset +1 (matches Table 2 in the paper).

    crash = At - Pre  (negative means confidence dropped at disambiguation)
    recovery = Post - At  (positive means it bounced back)
    """
    pre = conf_by_offset.get(-1)
    at = conf_by_offset.get(0)
    post = conf_by_offset.get(1)

    if pre is None or at is None or post is None:
        raise ValueError("missing pre/at/post offset - not enough data around disambiguation point")

    crash = at - pre
    recovery = post - at

    if crash < 0 and recovery > 0 and post < pre * 0.98:
        pattern = "Poisoning"       # crashed and didn't fully bounce back
    elif crash < 0 and post >= pre * 0.98:
        pattern = "Full Recovery"
    elif crash > 0 and recovery < 0:
        pattern = "Spike->Drop"     # extra attention at the hard point, then falls off
    else:
        pattern = "Partial Rec."

    return {"pre": pre, "at": at, "post": post, "crash": crash, "recovery": recovery, "pattern": pattern}
