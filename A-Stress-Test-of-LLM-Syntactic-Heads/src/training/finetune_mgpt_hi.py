"""Fine-tune mGPT (1.3B) on CC-100 Hindi. mGPT's SentencePiece tokenizer
handles Devanagari out of the box, which is why we used it instead of GPT-2
for the Hindi side - GPT-2's BPE vocab basically can't represent Hindi text.
"""
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

N_SENTENCES = 500_000
OUTPUT_DIR = "./mgpt_hi_finetuned"


def main():
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/mGPT")
    model = AutoModelForCausalLM.from_pretrained("sberbank-ai/mGPT")

    # CC-100 is huge, so stream it and just take the first 500k Hindi sentences
    # rather than downloading the whole thing
    raw = load_dataset("cc100", lang="hi", split="train", streaming=True)
    texts = [ex["text"] for i, ex in enumerate(raw) if i < N_SENTENCES]
    dataset = Dataset.from_dict({"text": texts}).train_test_split(test_size=0.01)

    def tokenize_hi(ex):
        return tokenizer(ex["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize_hi, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir="./mgpt_hi",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=1000,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
