"""
Pulls garden-path / center-embedded sentences out of a UD CoNLL-U treebank.

No manual annotation - we just look for known ambiguity-inducing dependency
patterns and flag the token where disambiguation happens. Works the same way
for English and Hindi since we're matching on UD relations, not surface word
order.

Patterns we flag:
  - reduced relative clause:  NP [V-ed] ... with no overt "that/which" (acl
    child of a noun with no preceding `mark`, main-verb POS is VERB not AUX)
  - NP/Z ambiguity: a clause-initial subordinate clause with no complementizer
    that could be misread as the main clause's object
  - center-embedding: a `csubj`/`acl:relcl` nested inside another clause's
    subject span, forcing the parser to hold state before hitting the matrix
    verb

This is a heuristic, not a gold-standard classifier - it's meant to surface
candidate sentences at the rate a psycholinguist would construct them by
hand, not to perfectly replicate any specific published garden-path corpus.
"""
import argparse
import json

from conllu import parse_incr


REDUCED_RC_DEPREL = {"acl", "acl:relcl"}
CLAUSAL_SUBJ = {"csubj", "csubj:pass"}


def has_overt_marker(token, sentence_by_id):
    """Check if a clausal child has an overt complementizer/relativizer (mark, PRON with case=rel)."""
    for tok in sentence_by_id.values():
        if tok["head"] == token["id"] and tok["deprel"] in ("mark", "nsubj:rel"):
            return True
    return False


def find_reduced_relatives(sentence):
    by_id = {t["id"]: t for t in sentence if isinstance(t["id"], int)}
    hits = []
    for tok in sentence:
        if not isinstance(tok["id"], int):
            continue
        if tok["deprel"] not in REDUCED_RC_DEPREL:
            continue
        head = by_id.get(tok["head"])
        if head is None or head["upos"] != "NOUN":
            continue
        if tok["upos"] != "VERB":
            continue
        if has_overt_marker(tok, by_id):
            continue
        # disambiguation point = the ambiguous verb itself, since that's where
        # the reader realizes it's not the matrix verb
        hits.append({"pattern": "reduced_relative", "disambig_id": tok["id"]})
    return hits


def find_npz_ambiguity(sentence):
    """Subordinate clause with no complementizer at the start of the sentence,
    e.g. 'While the man hunted the deer ran into the woods.'"""
    by_id = {t["id"]: t for t in sentence if isinstance(t["id"], int)}
    hits = []
    for tok in sentence:
        if not isinstance(tok["id"], int):
            continue
        if tok["deprel"] != "advcl":
            continue
        if tok["id"] > 4:  # only care about clause-initial subordinate clauses
            continue
        if has_overt_marker(tok, by_id):
            continue
        # find the object of the subordinate verb - that's the disambiguating
        # token once the real matrix subject shows up right after it
        obj = next((t for t in sentence if t["head"] == tok["id"] and t["deprel"] == "obj"), None)
        if obj is not None:
            hits.append({"pattern": "npz_ambiguity", "disambig_id": obj["id"]})
    return hits


def find_center_embedding(sentence):
    by_id = {t["id"]: t for t in sentence if isinstance(t["id"], int)}
    hits = []
    for tok in sentence:
        if not isinstance(tok["id"], int):
            continue
        if tok["deprel"] not in CLAUSAL_SUBJ:
            continue
        head = by_id.get(tok["head"])
        if head is None:
            continue
        # the matrix verb is the disambiguation point - reader has to pop the
        # embedded clause off the stack to get here
        hits.append({"pattern": "center_embedding", "disambig_id": head["id"]})
    return hits


PATTERN_FUNCS = [find_reduced_relatives, find_npz_ambiguity, find_center_embedding]


def extract(treebank_path, lang):
    out = []
    with open(treebank_path, "r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            text = sentence.metadata.get("text", "")
            for fn in PATTERN_FUNCS:
                for hit in fn(sentence):
                    out.append({
                        "lang": lang,
                        "text": text,
                        "sent_id": sentence.metadata.get("sent_id"),
                        "pattern": hit["pattern"],
                        "disambig_token_id": hit["disambig_id"],
                        "tokens": [t["form"] for t in sentence if isinstance(t["id"], int)],
                        "deps": [
                            {"id": t["id"], "form": t["form"], "head": t["head"], "deprel": t["deprel"], "upos": t["upos"]}
                            for t in sentence if isinstance(t["id"], int)
                        ],
                    })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--treebank", required=True, help="path to a .conllu file")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--lang", required=True, choices=["en", "hi"])
    args = ap.parse_args()

    records = extract(args.treebank, args.lang)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"found {len(records)} candidate sentences in {args.treebank} -> {args.out}")


if __name__ == "__main__":
    main()
