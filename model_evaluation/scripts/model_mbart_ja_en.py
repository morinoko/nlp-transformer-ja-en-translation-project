"""
Evaluation for base model
    ken11/mbart-ja-en
    https://huggingface.co/ken11/mbart-ja-en
"""
import os
import pandas as pd
import time
from transformers import MBartForConditionalGeneration, MBartTokenizer
import evaluate
from datasets import load_dataset

# Set up model
model = MBartForConditionalGeneration.from_pretrained("ken11/mbart-ja-en")

# setup tokenizer
tokenizer = MBartTokenizer.from_pretrained("ken11/mbart-ja-en")

# Set up evaluators
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")

print("####################################################")
print("Evaluation for ken11/mbart-ja-en")
print("####################################################")
print()

###################################################################
# Evaluate ken11/mbart-ja-en model on Tatoeba samples
###################################################################
print("-----------------------------------------")
print("Tatoeba samples")
print("-----------------------------------------")
print()

# format: {'id': '208864', 'translation': {'en': 'Tom loved studying French.', 'ja': 'トムはフランス語を勉強することが好きだった。'}}
tatoeba_dataset = load_dataset("tatoeba", lang1="en", lang2="ja", split="train")
shuffled_tatoeba_dataset = tatoeba_dataset.shuffle(seed=98)

# Use only a small portion of the data set due to computing power constrains
truncated_tatoeba_dataset = shuffled_tatoeba_dataset['translation'][:100]

en_tatoeba_references = [item['en'] for item in truncated_tatoeba_dataset]
ja_tatoeba_sequences = [item['ja'] for item in truncated_tatoeba_dataset]

print("Starting translation on Tatoeba samples...")
start = time.time()
encoded_ja_tatoeba = tokenizer(ja_tatoeba_sequences, return_tensors="pt", padding=True, truncation='longest_first')
generated_tatoeba_tokens = model.generate(**encoded_ja_tatoeba, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"], early_stopping=True, max_length=48)
tatoeba_output = tokenizer.batch_decode(generated_tatoeba_tokens, skip_special_tokens=True)
end = time.time()
print("Finished Tatoeba samples!")
print(f"Total translation time: {end - start:.4f} seconds")
print()

# Prepare evaluation
en_tatoeba_predictions = tatoeba_output
tatoeba_bleu = bleu.compute(predictions=en_tatoeba_predictions, references=en_tatoeba_references)['bleu']
tatoeba_chrf = chrf.compute(predictions=en_tatoeba_predictions, references=en_tatoeba_references)['score']

print("-----------------------------------------")
print("Tatoeba Evaluation scores")
print("-----------------------------------------")
print("BLEU - Tatoeba:", tatoeba_bleu)
print("chrF - Tatoeba:", tatoeba_chrf)
print()

# Show sample of translations
print("-----------------------------------------")
print("Translation samples from Tatoeba dataset:")
print("-----------------------------------------")
sample_size = 10

for i in range(sample_size):
  print(i + 1)
  print(f"Original JA: {ja_tatoeba_sequences[i]}")
  print(f"EN Prediction: {en_tatoeba_predictions[i]}")
  print(f"EN Reference: {en_tatoeba_references[i]}")
  print()

###################################################################
# Evaluate facebook/mbart-large-50-many-to-one-mmt on KFTT samples
###################################################################
print("-----------------------------------------")
print("KFTT samples")
print("-----------------------------------------")
print()

kftt_dataset = load_dataset("Hoshikuzu/KFTT", split="train")
kftt_shuffled_dataset = kftt_dataset.shuffle(seed=98)

# Use only a small portion of the data set for efficiency
kftt_truncated_dataset = kftt_shuffled_dataset['translation'][:100]

en_kftt_references = [item['en'] for item in kftt_truncated_dataset]
ja_kftt_sequences = [item['ja'] for item in kftt_truncated_dataset]

print("Starting translation on KFTT samples...")
start = time.time()
encoded_ja_kftt = tokenizer(ja_kftt_sequences, return_tensors="pt", padding=True, truncation='longest_first')
generated_kftt_tokens = model.generate(**encoded_ja_kftt, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"], early_stopping=True, max_length=48)
kftt_output = tokenizer.batch_decode(generated_kftt_tokens, skip_special_tokens=True)
end = time.time()
print("Finished KFTT samples!")
print(f"Total translation time: {end - start:.4f} seconds")
print()

# Prepare evaluation
en_kftt_predictions = kftt_output
kftt_bleu = bleu.compute(predictions=en_kftt_predictions, references=en_kftt_references)['bleu']
kftt_chrf = chrf.compute(predictions=en_kftt_predictions, references=en_kftt_references)['score']

print("-----------------------------------------")
print("KFTT Evaluation scores")
print("-----------------------------------------")
print("BLEU - KFTT:", kftt_bleu)
print("chrF - KFTT:", kftt_chrf)
print()

# Show sample of translations
print("-----------------------------------------")
print("Translation samples from KFTT dataset:")
print("-----------------------------------------")
sample_size = 10

for i in range(sample_size):
  print(i + 1)
  print(f"Original JA: {ja_kftt_sequences[i]}")
  print(f"EN Prediction: {en_kftt_predictions[i]}")
  print(f"EN Reference: {en_kftt_references[i]}")
  print()

#############################################################################################
# Evaluate facebook/mbart-large-50-many-to-one-mmt on custom Ruby programming documentation samples
#############################################################################################
print("-----------------------------------------")
print("Ruby documentation samples")
print("-----------------------------------------")
print()
ruby_dataset = load_dataset("morinoko-inari/ruby-rails-ja-en", split="train")

ruby_shuffled_dataset = ruby_dataset.shuffle(seed=98)
ruby_truncated_dataset = ruby_shuffled_dataset[:100]

en_ruby_references = ruby_truncated_dataset['en']
ja_ruby_sequences = ruby_truncated_dataset['ja']

print("Starting translation on Ruby samples...")
start = time.time()
encoded_ja_ruby = tokenizer(ja_ruby_sequences, return_tensors="pt", padding=True, truncation='longest_first')
generated_ruby_tokens = model.generate(**encoded_ja_ruby, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"], early_stopping=True, max_length=48)
ruby_output = tokenizer.batch_decode(generated_ruby_tokens, skip_special_tokens=True)
end = time.time()
print("Finished Ruby samples!")
print(f"Total translation time: {end - start:.4f} seconds")
print()

# Prepare evaluation
en_ruby_predictions = ruby_output
ruby_bleu = bleu.compute(predictions=en_ruby_predictions, references=en_ruby_references)['bleu']
ruby_chrf = chrf.compute(predictions=en_ruby_predictions, references=en_ruby_references)['score']

print("-----------------------------------------")
print("Ruby Evaluation scores")
print("-----------------------------------------")
print("BLEU - Ruby:", ruby_bleu)
print("chrF - Ruby:", ruby_chrf)
print()

# Show sample of translations
print("-----------------------------------------")
print("Translation samples from Ruby dataset:")
print("-----------------------------------------")
sample_size = 10

for i in range(sample_size):
  print(i + 1)
  print(f"Original JA: {ja_ruby_sequences[i]}")
  print(f"EN Prediction: {en_ruby_predictions[i]}")
  print(f"EN Reference: {en_ruby_references[i]}")
  print()

# Create Data CSV
csv_data = {
  'original_ja': ja_tatoeba_sequences + ja_kftt_sequences + ja_ruby_sequences,
  'en_predictions': en_tatoeba_predictions + en_kftt_predictions + en_ruby_predictions,
  'en_references': en_tatoeba_references + en_kftt_references + en_ruby_references,
  'dataset': (['tatoeba'] * len(ja_tatoeba_sequences)) + (['kftt'] * len(ja_kftt_sequences)) + (['ruby'] * len(en_ruby_predictions)),
}
df = pd.DataFrame(csv_data)

# Define the relative path for output file and make sure directory exists
output_dir = os.path.join(os.path.dirname(__file__), '..', 'translation_output_csvs')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'results_mbart.csv')
# Save CSV
df.to_csv(output_file, index=False)
