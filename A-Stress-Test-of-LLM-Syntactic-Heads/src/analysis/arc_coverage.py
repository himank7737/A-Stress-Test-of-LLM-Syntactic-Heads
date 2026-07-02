"""Fix 1: Bidirectional Arc Coverage.

A decoder-only model's causal mask means a token can only attend to tokens
before it. If the UD head comes after the dependent in the sentence (a
"right-pointing" arc), the dependent literally cannot attend to its head -
the attention weight will always be ~0 regardless of whether the model
"knows" the relation. Measured on our data: this blocks 63% of English UD
arcs and 94.6% of Hindi arcs, which would make Hindi numbers meaningless
without a fix.

The fix: for right-pointing arcs, we flip which token is doing the attending
- instead of asking "does the dependent attend to the head," we ask "does
the head attend back to the dependent." Same underlying relation, just
checked from the direction the causal mask actually allows.
"""


def is_right_pointing(dep_pos, head_pos):
    return head_pos > dep_pos


def arc_attention_weight(attn, dep_pos, head_pos):
    """attn: tensor of shape (n_heads, seq_len, seq_len) or (layers, heads, seq, seq).
    Returns the mean attention weight along the correct (causal-mask-legal)
    direction for this arc.
    """
    if is_right_pointing(dep_pos, head_pos):
        # head attends back to dependent
        src, tgt = head_pos, dep_pos
    else:
        # standard: dependent attends to head
        src, tgt = dep_pos, head_pos
    return float(attn[..., src, tgt].mean())


def arc_coverage_stats(sentences):
    """Diagnostic: what fraction of arcs would be structurally invisible to a
    causal model without this fix. Used to reproduce the 63% / 94.6% figures.
    """
    total, blocked = 0, 0
    for sent in sentences:
        for tok in sent["deps"]:
            if tok["head"] == 0:
                continue
            dep_pos = tok["id"]
            head_pos = tok["head"]
            total += 1
            if is_right_pointing(dep_pos, head_pos):
                blocked += 1
    return {"total_arcs": total, "blocked_arcs": blocked, "pct_blocked": 100 * blocked / total if total else 0}
