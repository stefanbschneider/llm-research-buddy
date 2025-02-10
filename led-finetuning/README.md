---
license: apache-2.0
datasets:
- stefanbschneider/lfqa-max-answer-length-512
language:
- en
base_model:
- allenai/led-base-16384
---


# Longformer Encoder-Decoder (LED) Fine-tuned for Generative Q&A

This model uses [Allenai's Longformer Encoder-Decoder (LED)](https://huggingface.co/docs/transformers/en/model_doc/led) as base model,
which supports very long contexts.

I fine-tuned the base model on [a long-form question answering (LFQA) dataset](https://huggingface.co/datasets/stefanbschneider/lfqa-max-answer-length-512) to perform generative/abstractive question answering (Q&A).


## Fine-tuning for down-stream task

I used the script [`led-finetune-lfqa-train.py`](https://huggingface.co/stefanbschneider/led-base-16384-lfqa-ans-len-512/blob/main/led-finetune-lfqa-train.py) in this repo to fine-tune the model on a GTX 4070s Ti.
Due to limited resources, I only trained on 50% of the full training set for only one epoch.

**For details, see my blog post: [Fine-Tuning a Pre-Trained LLM](https://stefanbschneider.github.io/blog/posts/llm-fine-tuning/)**

[This notebook](https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing) shows how *led-base-16384* can effectively be fine-tuned on a different summarization task. 
I used it as starting point.

## Usage

To use the fine-tuned model, load it from the hub and format question and context as follows:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("stefanbschneider/led-base-16384-lfqa-ans-len-512")
model = AutoModelForSeq2SeqLM.from_pretrained("stefanbschneider/led-base-16384-lfqa-ans-len-512")

# Abstract from "Attention is all you need" by Vaswani et al.: https://arxiv.org/abs/1706.03762
context = """The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely. Experiments on two machine translation tasks show these models to
be superior in quality while being more parallelizable and requiring significantly
less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task...
"""
question = "What's a transformer'?"
input_text = f"question: {question} context: {abstract}"
tokens = tokenizer(input_text, return_tensors="pt")

answer = tokenizer.decode(model.generate(**tokens, max_length=512)[0], skip_special_tokens=True)
# Unfortunately, the answer is not great. Not sure why? Too little training?
# Open an issue/discussion if you have suggestions!
```
