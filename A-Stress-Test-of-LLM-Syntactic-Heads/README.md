# A Stress Test of LLM Syntactic Heads

Does the syntactic attention inside a language model actually hold up when the input gets hard to
predict? That's the question this project is trying to answer. We take garden-path sentences 
and check whether the attention heads that track dependency relations lose confidence when
surprisal spikes.

We run this across two model families   GPT-2 (117M) and Llama 3.2 3B  and two different languages
 English (SVO) and Hindi (SOV), so four experiments total.

Course project for CGS410.

**team:** Himank Khandelwal, Virender, Sanchit Hamane, Kshitij Pramod Ramrekkar

## TLDR of the result

All four experiments show a significant negative correlation between token surprisal and syntactic
head confidence (p < .001 everywhere). GPT-2 has a much stronger coupling (r ≈ -0.40) than Llama
(r ≈ -0.14), which is consistent with larger models spreading syntactic information across more
heads instead of concentrating it. GPT-2 also crashes at the disambiguation point and doesn't fully
recover , while Llama-English actually spikes at the disambiguation word
before dropping - looks like it's throwing extra compute at the hard part rather than just
breaking.

Full writeup with all the figures is in `paper/`.

## Fixes We Did

The obvious approach   just grab attention weights and see if they line up with the gold UD arc  
runs into a few problems we had to explicitly correct for:

1. **Causal masking kills most of the arcs.** GPT-2 (and any decoder-only model) can only attend
   backwards. That means it structurally cannot attend to  63% of English UD arcs and  94.6% of
   Hindi ones, because the dependent often comes before the head in the sentence. If you don't fix
   this, you're mostly measuring  can the model see this token at all, not syntax. We fix it by
   flipping the query direction for right-pointing arcs (see `src/analysis/arc_coverage.py`).

2. **Not every head is a syntax head.** Most attention heads are doing something else entirely
   (positional stuff, previous-token heads, whatever). We isolate the heads that actually track UD
   structure above a random baseline + 2σ before computing anything (`head_identification.py`).
   This drops GPT 2 down to 10/144 heads and Llama down to 10-18/672.

3. **Root tokens are structurally different from everything else** and they were dragging the
   correlation around. We residualize both surprisal and confidence on `is_root` via OLS before
   computing Pearson r / Spearman ρ (`residualization.py`).

## Repo layout

```
src/
  data_prep/        pulls garden-path / center-embedded sentences out of UD CoNLL-U files
  training/         fine-tuning scripts (GPT-2 on WikiText-103, mGPT on CC-100 Hindi) + Llama loader
  analysis/         surprisal extraction, the three fixes above, correlation + recovery analysis
  pipeline/         run_experiment.py, glues everything together for one (model, language) run
  viz/              regenerates Figures 1-5 from the report
configs/            one yaml per experiment
```

You'll need a Hugging Face token with access to `meta-llama/Llama-3.2-3B` for the Llama
experiments. Export it before running anything:

```bash
export HF_TOKEN=your_token_here
```

GPU is basically mandatory   Llama-3.2-3B is loaded 4-bit quantized but you still want a GPU
with at least  8GB free. GPT-2 / mGPT fine-tuning will run on CPU but it'll take forever.

## Fine-tuning the base models

```bash
python -m src.training.finetune_gpt2_en
python -m src.training.finetune_mgpt_hi
```

Llama isn't fine-tuned   we use it with native weights for both languages (see `load_llama.py`),

## Building the garden-path dataset

We didn't hand-annotate anything. `extract_garden_path.py` walks a CoNLL-U treebank and flags
sentences matching known ambiguity patterns - same pipeline for
both English and Hindi, no manual labeling needed.

```bash
python -m src.data_prep.extract_garden_path \
    --treebank data/en_ewt-ud-train.conllu \
    --out data/garden_path_en.jsonl \
    --lang en
```

Treebanks used:
- English: [UD English Web Treebank](https://universaldependencies.org/) 
- Hindi: [Hindi Dependency Treebank](https://universaldependencies.org/) 

## Running an experiment

```bash
python -m src.pipeline.run_experiment --config configs/gpt2_en.yaml
```

This extracts surprisal + attention for every sentence, applies the three fixes, computes the
surprisal-confidence correlation (H1), runs the crash/recovery analysis around the disambiguation
token (H2), and dumps everything to `results/<experiment_name>/`.

## Regenerating the figures

```bash
python -m src.viz.plots --results results/gpt2_en --out figures/
```

## Limitations (being upfront about these)

- The bidirectional arc fix is a patch, not a real fix  it's still asymmetric compared to what a
  true bidirectional encoder like BERT would give you.
- Attention weight is a proxy for "syntactic processing," not the thing itself. High attention to
  the right token doesn't prove the model is using that information downstream.
- GPT 2 and Llama use different tokenizers and were trained on very different corpora, so the
  model size comparison is confounded with pretraining data differences. We can't fully separate
  "bigger model" from "different training run."
- The Hindi Llama numbers use the base multilingual model with no Hindi-specific fine tuning, so
  it's not an apples to apples comparison with the Hindi GPT-2 (mGPT) run, which was fine-tuned.

## References

- Clark et al. (2019), *What does BERT look at?*, BlackboxNLP
- Hale (2001), *A probabilistic Earley parser as a psycholinguistic model*, NAACL
- Levy (2008), *Expectation-based syntactic comprehension*, Cognition
- Tenney et al. (2019), *BERT rediscovers the classical NLP pipeline*, ACL
- Nivre et al. (2020), *Universal Dependencies v2*, LREC
