# Project Report and Presentation Slides

This directory contains the project report and presentation slides. The report includes detailed information about the models, datasets, evaluation metrics, methodology, and results. The presentation slides summarize the report, including key findings and insights from the project.

A video of the presentation is also available on YouTube: https://www.youtube.com/watch?v=TeqbkwpSctI

## Project Abstract

For machine translation, transformer-based models have been shown to perform better than RNNs and LSTM architectures. However, AI models often struggle with translating non-European and minor languages, especially when dealing with idiomatic expressions, niche topics, or vastly different grammar systems. In this paper, I look specifically at the Japanese-English language pair. First, I evaluate the performance of current open-source transformer models, in particular smaller models, in translating from Japanese to English. Second, I experiment with how fine-tuning a transformer model using a specialized, custom dataset could improve its performance for a specific topic. I also look into how the number of training samples provided during fine-tuning affect the fine-tuning results. The models were scored using BLEU and chrF metrics and training through the Hugging Face transformers library. The results suggest that the use of a curated dataset for fine-tuning could improve machine translation quality for a specific topic and that model performance generally increases with the volume of training data.
