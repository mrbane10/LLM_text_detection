# LLM-Generated Text Detection

This repository contains implementations of advanced NLP techniques aimed at detecting AI-generated text. The methods were developed as part of the [Kaggle Competition] in December 2023. The repository includes three key notebooks that explore various approaches to tackle this challenge.

---

## Table of Contents

1. [Overview](#overview)
2. [Notebooks](#notebooks)
    - [AI Detection using BLOOM](#ai-detection-using-bloom)
    - [Detect AI using LoRA-Finetuned Method](#detect-ai-using-lora-finetuned-method)
    - [Detect Using Model Ensemble](#detect-using-model-ensemble)
3. [Performance Highlights](#performance-highlights)

---

## Overview

Detecting AI-generated text is a crucial task in ensuring transparency and authenticity in natural language processing applications. In this repository, we apply a range of techniques, including tokenization, vectorization, fine-tuning, and ensemble modeling, to develop robust detection methods. 

Key highlights:
- Implemented **Byte Pair Encoding (BPE)** with **TF-IDF vectorization**.
- Built ensemble models using **CatBoost**, **LightGBM**, and **Naive Bayes**.
- Fine-tuned **DistilBERT** using **Low Rank Adaptation (LoRA)** for classification.
- Achieved a leaderboard score of **0.96** with ensemble models and **0.86** with the BERT-based approach.

---

## Notebooks

### 1. AI Detection using BLOOM

This notebook explores the use of **BLOOM**, a large language model, for detecting AI-generated text. It leverages tokenization and embeddings to classify text as either human-written or AI-generated. 

---

### 2. Detect AI using LoRA-Finetuned Method

This approach focuses on fine-tuning **DistilBERT** using **LoRA (Low Rank Adaptation)** for text classification. The method emphasizes reducing the computational overhead while maintaining high accuracy in detection.

---

### 3. Detect Using Model Ensemble

This notebook implements an ensemble model strategy, combining **CatBoost**, **LightGBM**, and **Naive Bayes**. It integrates **BPE** and **TF-IDF vectorization** for feature engineering, yielding superior performance on the leaderboard.

---

## Performance Highlights

| **Method**              | **Technique**        | **Score** |
|--------------------------|----------------------|-----------|
| Ensemble Model           | BPE + TF-IDF        | 0.96      |
| LoRA-Finetuned DistilBERT| Fine-tuning         | 0.86      |

---
