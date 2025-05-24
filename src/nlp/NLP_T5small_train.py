# from datasets import load_dataset
# from transformers import AutoTokenizer
# from transformers import DataCollatorForSeq2Seq
# import evaluate
# import numpy as np

# from transformers import AdamWeightDecay
# from transformers import TFAutoModelForSeq2SeqLM

# import tensorflow as tf
# from transformers.keras_callbacks import KerasMetricCallback
# from transformers.keras_callbacks import PushToHubCallback

# #dataset loading
# books = load_dataset("opus_books", "en-fr")
# books = books["train"].train_test_split(test_size=0.2)

# #tokenizer
# checkpoint = "google-t5/t5-small"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# source_lang = "en"
# target_lang = "fr" #has to be set to en for our problem
# prefix = "translate English to French: " #edit this to our problem


# def preprocess_function(examples):
#     inputs = [prefix + example[source_lang] for example in examples["translation"]]
#     targets = [example[target_lang] for example in examples["translation"]]
#     model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
#     return model_inputs


# tokenized_books = books.map(preprocess_function, batched=True)
# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")


# metric = evaluate.load("sacrebleu")

# def postprocess_text(preds, labels):
#     preds = [pred.strip() for pred in preds]
#     labels = [[label.strip()] for label in labels]

#     return preds, labels

# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     if isinstance(preds, tuple):
#         preds = preds[0]
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     result = {"bleu": result["score"]}

#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
#     result["gen_len"] = np.mean(prediction_lens)
#     result = {k: round(v, 4) for k, v in result.items()}
#     return result

# optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
# model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# tf_train_set = model.prepare_tf_dataset(
#     tokenized_books["train"],
#     shuffle=True,
#     batch_size=16,
#     collate_fn=data_collator,
# )

# tf_test_set = model.prepare_tf_dataset(
#     tokenized_books["test"],
#     shuffle=False,
#     batch_size=16,
#     collate_fn=data_collator,
# )

# model.compile(optimizer=optimizer)  # No loss argument!

# metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_test_set)

# push_to_hub_callback = PushToHubCallback(
#     output_dir="my_awesome_opus_books_model",
#     tokenizer=tokenizer,
# )


# callbacks = [metric_callback, push_to_hub_callback]

# model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)



# # -----------------------------------------------Inference
# from transformers import pipeline
# from transformers import AutoTokenizer
# from transformers import TFAutoModelForSeq2SeqLM

# text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."

# # Change `xx` to the language of the input and `yy` to the language of the desired output.
# # Examples: "en" for English, "fr" for French, "de" for German, "es" for Spanish, "zh" for Chinese, etc; translation_en_to_fr translates English to French
# # You can view all the lists of languages here - https://huggingface.co/languages
# translator = pipeline("translation_xx_to_yy", model="username/my_awesome_opus_books_model")
# translator(text)
# [{'translation_text': 'Legumes partagent des ressources avec des bactéries azotantes.'}]

# tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_opus_books_model")
# inputs = tokenizer(text, return_tensors="pt").input_ids

# model = TFAutoModelForSeq2SeqLM.from_pretrained("username/my_awesome_opus_books_model")
# outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)

# tokenizer.decode(outputs[0], skip_special_tokens=True)
# 'Les lignées partagent des ressources avec des bactéries enfixant l\'azote.'
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AdamWeightDecay, TFAutoModelForSeq2SeqLM
import evaluate
import numpy as np
import tensorflow as tf
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback
from os.path import join
import os

# Load dataset
books = load_dataset("opus_books", "en-fr")
books = books["train"].train_test_split(test_size=0.2)

# Tokenizer
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

tokenized_books = books.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# Model training
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)

tf_train_set = model.prepare_tf_dataset(
    tokenized_books["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    tokenized_books["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

model.compile(optimizer=optimizer)

# Callbacks (remove push_to_hub if not needed)
output_dir = "saved_models/my_awesome_opus_books_model"
os.makedirs(output_dir, exist_ok=True)

metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_test_set)
# PushToHubCallback commented unless you really want to push to huggingface
# push_to_hub_callback = PushToHubCallback(output_dir=output_dir, tokenizer=tokenizer)

callbacks = [metric_callback]  # add push_to_hub_callback if you want

model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)

# Save locally
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# ----------------------------------------------- Inference

from transformers import pipeline, AutoTokenizer, TFAutoModelForSeq2SeqLM

text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."

model_path = output_dir

translator = pipeline("translation_en_to_fr", model=model_path, tokenizer=model_path, framework="tf")
result = translator(text)
print(result)

