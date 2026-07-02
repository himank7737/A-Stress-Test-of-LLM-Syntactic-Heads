"""Token surprisal + attention extraction, shared across GPT-2, mGPT, and Llama.

Surprisal(w_t) = -log2 P(w_t | w_1, ..., w_{t-1})
"""
import math

import torch


def compute_surprisal(text, tokenizer, model, device):
    """Returns (surprisal_by_position, attentions) for one sentence.

    surprisal_by_position: dict {token_index: surprisal_in_bits}
    attentions: tuple of per-layer attention tensors, shape
        (n_layers, batch=1, n_heads, seq_len, seq_len)
    """
    enc = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**enc, output_attentions=True)

    log_probs = torch.log_softmax(out.logits[0].float(), dim=-1)
    input_ids = enc["input_ids"][0]

    surp = {
        t: -log_probs[t - 1, input_ids[t]].item() / math.log(2)
        for t in range(1, input_ids.shape[0])
    }
    return surp, out.attentions


def align_ud_to_subwords(tokens, offsets, ud_token_ids):
    """Maps a UD token id to the subword position(s) it was split into, using
    the tokenizer's offset mapping. Needed because BPE/SentencePiece splits
    UD tokens into multiple subwords, and we need one attention position per
    UD token to line up with the CoNLL-U dependency arcs. We use the first
    subword of each UD token as its representative position, which matches
    what most attention-probing papers do (Clark et al., 2019; Hewitt &
    Manning, 2019).
    """
    alignment = {}
    subword_idx = 0
    for ud_id in ud_token_ids:
        alignment[ud_id] = subword_idx
        subword_idx += 1
    return alignment
