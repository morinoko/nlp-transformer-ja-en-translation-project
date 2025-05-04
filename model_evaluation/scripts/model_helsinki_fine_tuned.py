"""
Evaluation for morinoko-inari/ruby-rails-fine-tuned-ja-en, a fine-tuned version
of Helsinki-NLP/opus-mt-ja-en using a custom dataset on the Ruby programming documentation.
    https://huggingface.co/morinoko-inari/ruby-rails-fine-tuned-ja-en
"""
import os
import pandas as pd
import time
from transformers import pipeline
import evaluate
from datasets import load_dataset

# Set up transformer pipeline for model
pipe = pipeline("translation", model="morinoko-inari/ruby-rails-fine-tuned-ja-en")

# Set up evaluators
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")

print("####################################################")
print("Evaluation for morinoko-inari/ruby-rails-fine-tuned-ja-en")
print("####################################################")
print()

###################################################################
# Evaluate morinoko-inari/ruby-rails-fine-tuned-ja-en model on Tatoeba samples
###################################################################
print("-----------------------------------------")
print("Tatoeba samples")
print("-----------------------------------------")
print()
# Prepare dataset
# dataset format: {'id': '208864', 'translation': {'en': 'Tom loved studying French.', 'ja': 'トムはフランス語を勉強することが好きだった。'}}
tatoeba_dataset = load_dataset("tatoeba", lang1="en", lang2="ja", split="train")
shuffled_tatoeba_dataset = tatoeba_dataset.shuffle(seed=98)

# Use only a small portion of the data set due to computing power constrains
truncated_tatoeba_dataset = shuffled_tatoeba_dataset['translation'][:100]

en_tatoeba_references = [item['en'] for item in truncated_tatoeba_dataset]
ja_tatoeba_sequences = [item['ja'] for item in truncated_tatoeba_dataset]

print("Starting translation on Tatoeba samples...")
start = time.time()
tatoeba_output = pipe(ja_tatoeba_sequences)
end = time.time()
print("Finished Tatoeba samples!")
print(f"Total translation time: {end - start:.4f} seconds")
print()

# Prepare evaluation
en_tatoeba_predictions = [item['translation_text'] for item in tatoeba_output]
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
# Evaluate morinoko-inari/ruby-rails-fine-tuned-ja-en model on KFTT samples
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
kftt_output = pipe(ja_kftt_sequences)
end = time.time()
print("Finished KFTT samples!")
print(f"Total translation time: {end - start:.4f} seconds")
print()

# Prepare evaluation
en_kftt_predictions = [item['translation_text'] for item in kftt_output]
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
# Evaluate morinoko-inari/ruby-rails-fine-tuned-ja-en model on custom Ruby programming documentation samples
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
ruby_output = pipe(ja_ruby_sequences)
end = time.time()
print("Finished Ruby samples!")
print(f"Total translation time: {end - start:.4f} seconds")
print()

# Prepare evaluation
en_ruby_predictions = [item['translation_text'] for item in ruby_output]
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
output_file = os.path.join(output_dir, 'results_helsinki_fine_tuned.csv')
# Save CSV
df.to_csv(output_file, index=False)
