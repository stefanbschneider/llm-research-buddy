# setup: pip install -U datasets evaluate transformers accelerate rouge_score wandb
from typing import Optional
from datasets import load_dataset
import evaluate
from transformers import (
    pipeline,
    AutoTokenizer,
)

# load model and tokenizer
model_name = "stefanbschneider/led-base-16384-lfqa-ans-len-512"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def concat_question_and_context(batch):
    # combine context strings and questions to one input
    batch["question_context"] = [
        f"question: {question}, context: {' '.join(context)}"
        for question, context in zip(batch["question"], batch["context"])
    ]
    return batch


def load_and_process_dataset(split: str, dataset_limit: Optional[int] = None):
    """Load and process the dataset for training or validation. Optionally limit the number of samples."""
    dataset = load_dataset("stefanbschneider/lfqa-max-answer-length-512", split=split)

    # optionally reduce the data sets to a small fraction
    if dataset_limit is not None:
        dataset = dataset.select(range(dataset_limit))

    dataset = dataset.map(
        concat_question_and_context,
        batched=True,
        batch_size=4,
        remove_columns=["context", "question"],
    )

    return dataset


# Load and process datasets
val_data = load_and_process_dataset("validation", dataset_limit=5)

# Prepare evaluation
rouge = evaluate.load("rouge")
pipe = pipeline("text2text-generation", model=model_name)
task_evaluator = evaluate.evaluator("text2text-generation")
results = task_evaluator.compute(
    model_or_pipeline=pipe,
    data=val_data,
    metric=rouge,
    input_column="question_context",
    label_column="answer",
)

evaluate.save("results/", **results)
print(results)
