import torch

from datasets import load_dataset
import evaluate
from transformers import LEDTokenizer, LEDForConditionalGeneration

# load pubmed
pubmed_test = load_dataset("scientific_papers", "pubmed", split="test")

# load tokenizer
tokenizer = LEDTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
model = LEDForConditionalGeneration.from_pretrained(
    "patrickvonplaten/led-large-16384-pubmed"
)


def generate_answer(batch):
    inputs_dict = tokenizer(
        batch["article"],
        padding="max_length",
        max_length=8192,
        return_tensors="pt",
        truncation=True,
    )
    input_ids = inputs_dict.input_ids
    attention_mask = inputs_dict.attention_mask
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    predicted_abstract_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
    )
    batch["predicted_abstract"] = tokenizer.batch_decode(
        predicted_abstract_ids, skip_special_tokens=True
    )
    return batch


result = pubmed_test.map(generate_answer, batched=True, batch_size=4)

# load rouge
rouge = evaluate.load("rouge")

print(
    "Result:",
    rouge.compute(
        predictions=result["predicted_abstract"],
        references=result["abstract"],
        rouge_types=["rouge2"],
    )["rouge2"],
)
