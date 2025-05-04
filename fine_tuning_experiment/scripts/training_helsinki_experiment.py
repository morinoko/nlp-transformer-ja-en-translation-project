from transformers import pipeline
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

"""
preprocess_function
This function tokenizes the input and target sequences.
"""
def preprocess_function(examples):
    # Tokenize the inputs and targets/labels
    inputs = [ex for ex in examples["ja"]]
    targets = [ex for ex in examples["en"]]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

"""
compute_metrics
This function computes the BLEU and chrF scores for the model predictions.
"""
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and references
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Evaluation library expects a list of references for each prediction
    references = [[ref] for ref in decoded_labels]
    bleu_result = bleu.compute(predictions=decoded_preds, references=references)['bleu']
    chrf_result = chrf.compute(predictions=decoded_preds, references=references)['score']

    return {"bleu": bleu_result, "chrf": chrf_result}

####################################################
# Set up evaluators
####################################################
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")

####################################################
# Prepare four datasets for training.
# Each set has an increasing number of training samples.
####################################################
print("Preparing datasets for training...")
dataset = load_dataset("morinoko-inari/ruby-rails-ja-en", split="train")
dataset_100 = dataset.train_test_split(train_size=100, test_size=50, shuffle=True, seed=42)
dataset_200 = dataset.train_test_split(train_size=200, test_size=50, shuffle=True, seed=42)
dataset_300 = dataset.train_test_split(train_size=300, test_size=50, shuffle=True, seed=42)
dataset_all = dataset.train_test_split(test_size=50, shuffle=True, seed=42)
print("Finished preparing datasets!")
print()

####################################################
# Set up models and tokenizer
####################################################
print("Preparing model and tokenizer...")
checkpoint = "Helsinki-NLP/opus-mt-ja-en"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print("Finished setting up model and tokenizer.")
print()

print("Tokenizing datasets...")
tokenized_datasets_100 = dataset_100.map(preprocess_function, batched=True)
tokenized_datasets_200 = dataset_200.map(preprocess_function, batched=True)
tokenized_datasets_300 = dataset_300.map(preprocess_function, batched=True)
tokenized_datasets_all = dataset_all.map(preprocess_function, batched=True)

print("Finished tokenizing datasets:")
print("100 samples:")
print(tokenized_datasets_100)
print()
print("200 samples:")
print(tokenized_datasets_200)
print()
print("300 samples:")
print(tokenized_datasets_300)
print()
print("All samples:")
print(tokenized_datasets_all)
print()


####################################################
# Set up training arguments and Data Collator
####################################################
training_args_100 = Seq2SeqTrainingArguments(
    output_dir="./training_results_100",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,  # Use mixed precision if you have compatible hardware
    push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
)
training_args_200 = Seq2SeqTrainingArguments(
    output_dir="./training_results_200",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,  # Use mixed precision if you have compatible hardware
    push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
)
training_args_300 = Seq2SeqTrainingArguments(
    output_dir="./training_results_300",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,  # Use mixed precision if you have compatible hardware
    push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
)
training_args_all = Seq2SeqTrainingArguments(
    output_dir="./training_results_all",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,  # Use mixed precision if you have compatible hardware
    push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


####################################################
# Train all models with different sample sizes
####################################################
print("Starting training with 100 samples...")
# Initialize the trainer and start training
trainer_100 = Seq2SeqTrainer(
    model=model,
    args=training_args_100,
    train_dataset=tokenized_datasets_100["train"],
    eval_dataset=tokenized_datasets_100["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer_100.train()

print("Finished training with 100 samples!")
print()

print("Saving 100 sample model...")
# Save the model locally
trainer_100.save_model("./ruby-rails-fine-tuned-ja-en_100")
print("Saved!")
print()


print("Starting training with 200 samples...")
# Initialize the trainer and start training
trainer_200 = Seq2SeqTrainer(
    model=model,
    args=training_args_200,
    train_dataset=tokenized_datasets_200["train"],
    eval_dataset=tokenized_datasets_200["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer_200.train()

print("Finished training with 200 samples!")
print()

print("Saving 200 sample model...")
# Save the model locally
trainer_200.save_model("./ruby-rails-fine-tuned-ja-en_200")
print("Saved!")
print()


print("Starting training with 300 samples...")
# Initialize the trainer and start training
trainer_300 = Seq2SeqTrainer(
    model=model,
    args=training_args_300,
    train_dataset=tokenized_datasets_300["train"],
    eval_dataset=tokenized_datasets_300["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer_300.train()

print("Finished training with 300 samples!")
print()

print("Saving 300 sample model...")
# Save the model locally
trainer_300.save_model("./ruby-rails-fine-tuned-ja-en_300")
print("Saved!")
print()


print("Starting training with all samples...")
# Initialize the trainer and start training
trainer_all = Seq2SeqTrainer(
    model=model,
    args=training_args_all,
    train_dataset=tokenized_datasets_all["train"],
    eval_dataset=tokenized_datasets_all["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer_all.train()
print("Finished training with all samples!")
print()

print("Saving all sample model...")
# Save the model locally
trainer_all.save_model("./ruby-rails-fine-tuned-ja-en_all")
print("Saved!")
print()

print("Finished training ALL the models!")

####################################################
# Evaluate all models with the test set
####################################################
print("Starting fine-tuned model evaluation with test set:")

# Load locally saved fine-tuned models
model_path_100 = ".ruby-rails-fine-tuned-ja-en_100"
fine_tuned_pipe_100 = pipeline("translation", model=model_path_100)

model_path_200 = "./ruby-rails-fine-tuned-ja-en_200"
fine_tuned_pipe_200 = pipeline("translation", model=model_path_200)

model_path_300 = "./ruby-rails-fine-tuned-ja-en_300"
fine_tuned_pipe_300 = pipeline("translation", model=model_path_300)

model_path_all = "./ruby-rails-fine-tuned-ja-en_all"
fine_tuned_pipe_all = pipeline("translation", model=model_path_all)

original_pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")


# Test with test set samples
test_dataset = dataset_all['test']
en_references = test_dataset['en']
ja_sequences = test_dataset['ja']

fine_tuned_result_100 = fine_tuned_pipe_100(ja_sequences)
fine_tuned_en_predictions_100 = [item['translation_text'] for item in fine_tuned_result_100]
fine_tuned_ruby_bleu_100 = bleu.compute(predictions=fine_tuned_en_predictions_100, references=en_references)['bleu']
fine_tuned_ruby_chrf_100 = chrf.compute(predictions=fine_tuned_en_predictions_100, references=en_references)['score']

fine_tuned_result_200 = fine_tuned_pipe_200(ja_sequences)
fine_tuned_en_predictions_200 = [item['translation_text'] for item in fine_tuned_result_200]
fine_tuned_ruby_bleu_200 = bleu.compute(predictions=fine_tuned_en_predictions_200, references=en_references)['bleu']
fine_tuned_ruby_chrf_200 = chrf.compute(predictions=fine_tuned_en_predictions_200, references=en_references)['score']

fine_tuned_result_300 = fine_tuned_pipe_300(ja_sequences)
fine_tuned_en_predictions_300 = [item['translation_text'] for item in fine_tuned_result_300]
fine_tuned_ruby_bleu_300 = bleu.compute(predictions=fine_tuned_en_predictions_300, references=en_references)['bleu']
fine_tuned_ruby_chrf_300 = chrf.compute(predictions=fine_tuned_en_predictions_300, references=en_references)['score']

fine_tuned_result_all = fine_tuned_pipe_all(ja_sequences)
fine_tuned_en_predictions_all = [item['translation_text'] for item in fine_tuned_result_all]
fine_tuned_ruby_bleu_all = bleu.compute(predictions=fine_tuned_en_predictions_all, references=en_references)['bleu']
fine_tuned_ruby_chrf_all = chrf.compute(predictions=fine_tuned_en_predictions_all, references=en_references)['score']

original_result = original_pipe(ja_sequences)
original_en_predictions = [item['translation_text'] for item in original_result]
original_ruby_bleu = bleu.compute(predictions=original_en_predictions, references=en_references)['bleu']
original_ruby_chrf = chrf.compute(predictions=original_en_predictions, references=en_references)['score']

print("-----------------------------------------")
print("Fine-tuned Model Evaluation scores vs original model")
print("-----------------------------------------")
print("BLEU - Original Ruby:", original_ruby_bleu)
print("chrF - Original Ruby:", original_ruby_chrf)
print("BLEU - Fine-tuned Ruby 100 samples:", fine_tuned_ruby_bleu_100)
print("chrF - Fine-tuned Ruby 100 samples:", fine_tuned_ruby_chrf_100)
print("BLEU - Fine-tuned Ruby 200 samples:", fine_tuned_ruby_bleu_200)
print("chrF - Fine-tuned Ruby 200 samples:", fine_tuned_ruby_chrf_200)
print("BLEU - Fine-tuned Ruby 300 samples:", fine_tuned_ruby_bleu_300)
print("chrF - Fine-tuned Ruby 300 samples:", fine_tuned_ruby_chrf_300)
print("BLEU - Fine-tuned Ruby all samples:", fine_tuned_ruby_bleu_all)
print("chrF - Fine-tuned Ruby all samples:", fine_tuned_ruby_chrf_all)
print()

print("-----------------------------------------")
print("Translations from fine-tuned model test data output compared to original:")
print("-----------------------------------------")
sample_size = len(test_dataset)

for i in range(sample_size):
    print(i + 1)
    print(f"Original JA: {ja_sequences[i]}")
    print(f"EN Reference: {en_references[i]}")
    print("---------------")
    print(f"Original EN Prediction: {original_en_predictions[i]}")
    print(f"Fine-tuned EN Prediction 100 samples: {fine_tuned_en_predictions_100[i]}")
    print(f"Fine-tuned EN Prediction 200 samples: {fine_tuned_en_predictions_200[i]}")
    print(f"Fine-tuned EN Prediction 300 samples: {fine_tuned_en_predictions_300[i]}")
    print(f"Fine-tuned EN Prediction all samples: {fine_tuned_en_predictions_all[i]}")
    print()
