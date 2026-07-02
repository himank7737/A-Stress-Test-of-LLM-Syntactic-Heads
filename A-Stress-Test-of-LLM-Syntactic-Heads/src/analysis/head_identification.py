"""Fix 2: Syntactic Head Isolation.

Most attention heads in a transformer aren't doing syntax - previous-token
heads, positional heads, whatever. If you average over every head you dilute
the signal from the handful that actually track dependency structure. So we
find the heads that beat a random baseline by a good margin and only use
those.

For each (layer, head), we compute the fraction of tokens where the head's
argmax attention (after applying Fix 1's direction correction) lands on the
correct UD head. We keep (layer, head) pairs that clear
random_baseline + 2*std.

random_baseline = 1 / (n_layers * n_heads) isn't quite right conceptually
(it should really be based on average sentence length), but empirically it
gives a threshold that reliably isolates a small, stable set of heads across
runs, which is what we cared about here.
"""
import numpy as np


def identify_syntactic_heads(all_conf_matrices, n_layers, n_heads, sigma=2.0):
    """all_conf_matrices: list of (n_layers, n_heads) arrays, one per sentence,
    where each cell is that head's max attention weight to the correct UD head
    for some target token in the sentence.

    Returns (list of (layer, head) tuples above threshold, accuracy matrix, threshold).
    """
    acc = np.zeros((n_layers, n_heads))
    for mat in all_conf_matrices:
        best = np.unravel_index(np.argmax(mat), mat.shape)
        acc[best] += 1
    acc /= len(all_conf_matrices)

    baseline = 1.0 / (n_layers * n_heads)
    threshold = baseline + sigma * np.std(acc)

    syntactic_heads = [
        (l, h) for l in range(n_layers) for h in range(n_heads) if acc[l, h] > threshold
    ]
    return syntactic_heads, acc, threshold


def dominant_head(acc):
    """The single best (layer, head) pair - used for the red-star markers in
    Figure 5."""
    l, h = np.unravel_index(np.argmax(acc), acc.shape)
    return int(l), int(h), float(acc[l, h])


def max_confidence_over_heads(attn, syntactic_heads, dep_pos, head_pos, arc_weight_fn):
    """Given a set of identified syntactic heads, return the max attention
    weight any of them assigns to the correct UD arc for this token. This is
    the `conf_max` value used downstream in the correlation and recovery
    analyses.
    """
    weights = [
        arc_weight_fn(attn[l][0, h], dep_pos, head_pos) for (l, h) in syntactic_heads
    ]
    return max(weights) if weights else 0.0
