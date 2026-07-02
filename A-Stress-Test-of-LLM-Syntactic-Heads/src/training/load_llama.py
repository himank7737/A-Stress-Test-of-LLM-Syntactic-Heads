"""Load Llama-3.2-3B in 4-bit (NF4) for both English and Hindi. Native
weights, no fine-tuning - we wanted to see what the model already encodes
rather than what we could teach it in a weekend on a single GPU.
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.2-3B"
N_LAYERS = 28
N_HEADS = 24
TOTAL_HEADS = N_LAYERS * N_HEADS  # 672


def load_llama(device_map="auto"):
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        raise RuntimeError("set HF_TOKEN env var - Llama-3.2-3B is a gated model")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        output_attentions=True,
        quantization_config=bnb_config,
        device_map=device_map,
        token=hf_token,
    )
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_llama()
    print(f"loaded {MODEL_ID}: {N_LAYERS} layers x {N_HEADS} heads = {TOTAL_HEADS} total heads")
