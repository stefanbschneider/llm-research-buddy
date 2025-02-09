# Very useful advice from BahnGPT: https://bahngpt.genai-prod.comp.db.de/chat/xCZIJYqvNwFjjXWusDUyeMU1PanKTUEYtpoI
from datetime import datetime
from typing import Optional
from datasets import load_dataset
import evaluate
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
)
import wandb


BATCH_SIZE: int = 2

# initialize wandb for monitoring
run_name: str = f"mac-m1_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
wandb.init(project="led-finetune-pubmed-summary", name=run_name)  # type: ignore

# load rouge for evaluation
rouge = evaluate.load("rouge")


# load model and tokenizer
# larger model for better performance, but slower to train: allenai/led-large-16384
pretrained_model_name = "allenai/led-base-16384"
my_model_name = "stefanbschneider/led-base-16384-pubmed-summary"
model_name = my_model_name
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# Enable gradient checkpointing to reduce memory during training (at the cost of speed)
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check tokenizer configuration
print(f"CLS Token ID: {tokenizer.cls_token_id}")
print(f"PAD Token ID: {tokenizer.pad_token_id}")
print(f"BOS Token ID: {tokenizer.bos_token_id}")
print(f"EOS Token ID: {tokenizer.eos_token_id}")
print(f"SEP Token ID: {tokenizer.sep_token_id}")
print(f"MASK Token ID: {tokenizer.mask_token_id}")
print(f"Additional Special Tokens: {tokenizer.additional_special_tokens}")


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        # Articles in PubMed are max 8k tokens long --> set max encoder length accordingly
        max_length=8192,
    )
    outputs = tokenizer(
        batch["abstract"],
        padding="max_length",
        truncation=True,
        # max decoder length determines the length of the answer
        # In PubMed, answers are the abstracts, which are max 512 tokens long
        max_length=512,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


def load_and_process_dataset(split: str, dataset_limit: Optional[int] = None):
    """Load and process the dataset for training or validation. Optionally limit the number of samples."""
    dataset = load_dataset(
        "scientific_papers", "pubmed", trust_remote_code=True, split=split
    )

    # comment out following lines for a full run
    # they reduce the data sets to a small fraction
    if dataset_limit is not None:
        dataset = dataset.select(range(dataset_limit))

    dataset = dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=["article", "abstract", "section_names"],
    )

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
    )

    return dataset


def compute_metrics(pred) -> dict[str, float]:
    """Compute rouge score during validation/evaluation"""
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"]

    # Return rouge2 F1 score
    # There are no longer separate precisoin, recall values in rouge2
    return {"rouge2": round(rouge_output, 4)}


### Start of actual script

# Load and process datasets
pubmed_train = load_and_process_dataset("train", dataset_limit=250)
pubmed_val = load_and_process_dataset("validation", dataset_limit=25)

# Create and set generation config
generation_config = GenerationConfig(
    # The generated answer/summary should be 100-512 tokens long
    max_length=512,
    min_length=100,
    early_stopping=True,
    num_beams=4,
    length_penalty=2.0,
    # Don't repeat n=3-grams (same words in same order) in the generated text --> more natural
    no_repeat_ngram_size=3,
    decoder_start_token_id=tokenizer.cls_token_id,
    bos_token_id=tokenizer.bos_token_id,
)
model.generation_config = generation_config

# Set training arguments
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="steps",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    # fp16 only works on GPU, not on M1 mps. mps is used by default if it's available
    # fp16=True,
    # fp16_backend="apex",
    output_dir=f"models/{my_model_name}",
    logging_steps=10,  # 50,
    eval_steps=50,  # 100,
    save_steps=20,  # 100,
    # warmup_steps=100,
    save_total_limit=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    # Save to HF hub & log to wandb
    push_to_hub=True,
    hub_model_id=my_model_name,
    log_level="info",
    report_to="wandb",
    run_name=run_name,
)

# start training
# Total steps = (num examples in data / (batch size * gradient accumulation steps)) * num epochs
# = eg, (32 / (1 * 1)) * 1 = 32
# The gradient accumulation adds multiple batches together before updating the weights
trainer = Seq2SeqTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=pubmed_train,
    eval_dataset=pubmed_val,
)
trainer.train() #resume_from_checkpoint=True)
trainer.push_to_hub()
