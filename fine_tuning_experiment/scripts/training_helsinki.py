from transformers import pipeline
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

def preprocess_function(examples):
    # Tokenize the inputs and labels
    inputs = [ex for ex in examples["ja"]]
    targets = [ex for ex in examples["en"]]

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

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
# Prepare dataset for training.
####################################################
print("Preparing dataset for training...")
dataset = load_dataset("morinoko-inari/ruby-rails-ja-en", split="train")
dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
print("Finished preparing dataset!")
print()

####################################################
# Set up model and tokenizer
####################################################
print("Preparing model and tokenizer...")
checkpoint = "Helsinki-NLP/opus-mt-ja-en"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print("Finished setting up model and tokenizer.")
print()

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
print("Finished tokenizing dataset:")
print(tokenized_datasets)
print()


####################################################
# Set up training arguments and Data Collator
####################################################
training_args = Seq2SeqTrainingArguments(
    output_dir="../../training-results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,  # Use mixed precision if there is compatible hardware available
    push_to_hub=False,  # Set to True to push to Hugging Face Hub
    hub_model_id="morinoko-inari/ruby-rails-fine-tuned-ja-en"  # The model name on the Hub
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

####################################################
# Train all models with different sample sizes
####################################################
print("Starting training...")

# Initialize the trainer and start training
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.train()

print("Finished training!")
print()

print("Saving model and uploading to Hugging Face Hub...")
# Save the model locally
trainer.save_model("./ruby-rails-fine-tuned-ja-en")

# Upload to Hugging Face Hub if logged in and set push_to_hub=True
# Note: To log in to the Hugging Face Hub: huggingface-cli login
if training_args.push_to_hub:
    trainer.push_to_hub()

print("Saved!")
print()
print("Finished training the model!")
print()


####################################################
# Model evaluation with test set
####################################################
print("Starting fine-tuned evaluation with test set:")
# Load your fine-tuned model
model_path = "morinoko-inari/ruby-rails-fine-tuned-ja-en"  # or local path "./my-fine-tuned-ja-en"
fine_tuned_pipe = pipeline("translation", model=model_path)
original_pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")

# Test with test set samples
test_dataset = dataset['test']
en_references = test_dataset['en']
ja_sequences = test_dataset['ja']

# Fine-tuned model results
fine_tuned_result = fine_tuned_pipe(ja_sequences)
fine_tuned_en_predictions = [item['translation_text'] for item in fine_tuned_result]
fine_tuned_ruby_bleu = bleu.compute(predictions=fine_tuned_en_predictions, references=en_references)['bleu']
fine_tuned_ruby_chrf = chrf.compute(predictions=fine_tuned_en_predictions, references=en_references)['score']

# Base/original model results
original_result = original_pipe(ja_sequences)
original_en_predictions = [item['translation_text'] for item in original_result]
original_ruby_bleu = bleu.compute(predictions=original_en_predictions, references=en_references)['bleu']
original_ruby_chrf = chrf.compute(predictions=original_en_predictions, references=en_references)['score']

print("-----------------------------------------")
print("Fine-tuned Model Evaluation scores vs original model")
print("-----------------------------------------")
print("BLEU - Fine-tuned Ruby:", fine_tuned_ruby_bleu)
print("chrF - Fine-tuned Ruby:", fine_tuned_ruby_chrf)
print("BLEU - Original Ruby:", original_ruby_bleu)
print("chrF - Original Ruby:", original_ruby_chrf)
print()

print("-----------------------------------------")
print("Translations from fine-tuned model test data output compared to original:")
print("-----------------------------------------")
sample_size = len(test_dataset)

for i in range(sample_size):
  print(i + 1)
  print(f"Original JA: {ja_sequences[i]}")
  print(f"Fine-tuned EN Prediction: {fine_tuned_en_predictions[i]}")
  print(f"Original EN Prediction: {original_en_predictions[i]}")
  print(f"EN Reference: {en_references[i]}")
  print()