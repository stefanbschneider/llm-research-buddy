---
library_name: transformers
license: apache-2.0
base_model: allenai/led-base-16384
tags:
- generated_from_trainer
- question_answering
- lfqa
- led
model-index:
- name: led-base-16384-lfqa-ans-len-512
  results: []
datasets:
- stefanbschneider/lfqa-max-answer-length-512
language:
- en
metrics:
- rouge
---


# Longformer Encoder-Decoder (LED) Fine-tuned for Generative Q&A

This model uses [Allenai's Longformer Encoder-Decoder (LED)](https://huggingface.co/docs/transformers/en/model_doc/led) as base model,
which supports very long contexts.
It is trained to generate answers to questions based on given contexts with answers of up to 512 tokens.

This model is a fine-tuned version of [allenai/led-base-16384](https://huggingface.co/allenai/led-base-16384) on the long-form question answering (LFQA) dataset [stefanbschneider/lfqa-max-answer-length-512](https://huggingface.co/datasets/stefanbschneider/lfqa-max-answer-length-512).

I used the script [`led-finetune-lfqa-train.py`](https://huggingface.co/stefanbschneider/led-base-16384-lfqa-ans-len-512/blob/main/led-finetune-lfqa-train.py) in this repo to fine-tune the model on a GTX 4070s Ti.


**For details, see my blog post: [Fine-Tuning a Pre-Trained LLM](https://stefanbschneider.github.io/blog/posts/llm-fine-tuning/)**

## Intended uses & limitations

Intended use: Generative/abstractive question answering with potentially very long contexts and multi-sentence answers.

Limitations: Limited training/fine-tuning, i.e., the model tends to ramble and the generated answers do not always make sense.

## Training and evaluation data

The model was fine-tuned on [stefanbschneider/lfqa-max-answer-length-512](https://huggingface.co/datasets/stefanbschneider/lfqa-max-answer-length-512).
Due to limited resources, I only trained on 50% of the full training set for only one epoch and performed evaluation on a small, fixed subset of the validation set.

It achieves the following results on the subset of the evaluation set:
- Loss: 3.2574
- Rouge2: 0.0416

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 1
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step  | Validation Loss | Rouge2 |
|:-------------:|:------:|:-----:|:---------------:|:------:|
| 3.6685        | 0.0197 | 1000  | 3.5648          | 0.0353 |
| 3.641         | 0.0395 | 2000  | 3.5259          | 0.0366 |
| 3.558         | 0.0592 | 3000  | 3.5224          | 0.0411 |
| 3.6013        | 0.0789 | 4000  | 3.4833          | 0.0327 |
| 3.5962        | 0.0986 | 5000  | 3.4795          | 0.0349 |
| 3.5325        | 0.1184 | 6000  | 3.4863          | 0.035  |
| 3.5618        | 0.1381 | 7000  | 3.4671          | 0.041  |
| 3.5344        | 0.1578 | 8000  | 3.4576          | 0.0339 |
| 3.515         | 0.1775 | 9000  | 3.4483          | 0.038  |
| 3.4672        | 0.1973 | 10000 | 3.4422          | 0.0343 |
| 3.448         | 0.2170 | 11000 | 3.4324          | 0.0369 |
| 3.5145        | 0.2367 | 12000 | 3.4304          | 0.0353 |
| 3.4565        | 0.2565 | 13000 | 3.4169          | 0.0382 |
| 3.4446        | 0.2762 | 14000 | 3.4061          | 0.0376 |
| 3.5298        | 0.2959 | 15000 | 3.3983          | 0.0368 |
| 3.459         | 0.3156 | 16000 | 3.3971          | 0.0387 |
| 3.4825        | 0.3354 | 17000 | 3.3985          | 0.04   |
| 3.3953        | 0.3551 | 18000 | 3.4034          | 0.0389 |
| 3.3849        | 0.3748 | 19000 | 3.3878          | 0.0345 |
| 3.4979        | 0.3945 | 20000 | 3.3890          | 0.038  |
| 3.4667        | 0.4143 | 21000 | 3.3744          | 0.0381 |
| 3.4154        | 0.4340 | 22000 | 3.3882          | 0.0376 |
| 3.4191        | 0.4537 | 23000 | 3.3585          | 0.0437 |
| 3.4372        | 0.4734 | 24000 | 3.3592          | 0.0395 |
| 3.4556        | 0.4932 | 25000 | 3.3557          | 0.0384 |
| 3.4234        | 0.5129 | 26000 | 3.3596          | 0.0386 |
| 3.413         | 0.5326 | 27000 | 3.3565          | 0.0329 |
| 3.3855        | 0.5524 | 28000 | 3.3475          | 0.0388 |
| 3.4496        | 0.5721 | 29000 | 3.3392          | 0.0372 |
| 3.4472        | 0.5918 | 30000 | 3.3332          | 0.0405 |
| 3.4109        | 0.6115 | 31000 | 3.3286          | 0.0413 |
| 3.4177        | 0.6313 | 32000 | 3.3194          | 0.046  |
| 3.4429        | 0.6510 | 33000 | 3.3043          | 0.0438 |
| 3.3835        | 0.6707 | 34000 | 3.2992          | 0.0411 |
| 3.4086        | 0.6904 | 35000 | 3.2984          | 0.04   |
| 3.4113        | 0.7102 | 36000 | 3.2973          | 0.0393 |
| 3.3986        | 0.7299 | 37000 | 3.2920          | 0.0418 |
| 3.3741        | 0.7496 | 38000 | 3.2915          | 0.0391 |
| 3.3473        | 0.7694 | 39000 | 3.2865          | 0.0434 |
| 3.3613        | 0.7891 | 40000 | 3.2776          | 0.0429 |
| 3.3411        | 0.8088 | 41000 | 3.2849          | 0.0385 |
| 3.2708        | 0.8285 | 42000 | 3.2760          | 0.0411 |
| 3.3755        | 0.8483 | 43000 | 3.2715          | 0.04   |
| 3.3551        | 0.8680 | 44000 | 3.2734          | 0.0363 |
| 3.3064        | 0.8877 | 45000 | 3.2678          | 0.0394 |
| 3.2962        | 0.9074 | 46000 | 3.2663          | 0.0434 |
| 3.2761        | 0.9272 | 47000 | 3.2658          | 0.0421 |
| 3.3495        | 0.9469 | 48000 | 3.2626          | 0.0433 |
| 3.3016        | 0.9666 | 49000 | 3.2600          | 0.0427 |
| 3.2545        | 0.9863 | 50000 | 3.2574          | 0.0416 |


### Framework versions

- Transformers 4.48.3
- Pytorch 2.5.1+cu121
- Datasets 3.2.0
- Tokenizers 0.21.0