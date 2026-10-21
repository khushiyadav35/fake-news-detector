# VeriFact: Fake News & WhatsApp Misinformation Detector with Explainable AI (LIME)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![XAI-LIME](https://img.shields.io/badge/Explainability-LIME-yellow.svg)](https://github.com/marcotcr/lime)

> **B.Tech Capstone / Final Year Project**  
> **Author:** Khushi Yadav ([@khushiyadav35](https://github.com/khushiyadav35))  
> **Domain:** Natural Language Processing (NLP), Explainable AI (XAI), Information Verification

---

## 📌 Problem Statement

In instant messaging platforms like WhatsApp and Telegram, viral misinformation, forward chains, medical hoaxes, and phishing scams proliferate at alarming speed. Unlike traditional published news, **WhatsApp forwards** possess distinct linguistic patterns:
- Hyperbolic claims and emotional manipulation
- Urgent calls to action (*"FORWARD TO 10 GROUPS BEFORE MIDNIGHT"*)
- Fabricated authority attributions (*"UNESCO declared...", "NASA confirmed..."*)
- High capitalization and excessive punctuation (`!!!`, `???`)

This project builds an end-to-end NLP classification pipeline that differentiates between **Reliable News** and **Suspicious/Misinformation Messages**, and leverages **LIME (Local Interpretable Model-agnostic Explanations)** to visually highlight the influential words driving each prediction.

---

## 🏗️ Project Architecture

```
fake-news-detector/
├── .github/workflows/    # CI pipeline for automated testing
├── app/                  # Streamlit web dashboard with interactive LIME visualizer
├── data/                 # Raw and processed benchmark datasets
├── models/               # Serialized models and TF-IDF vectorizer pipelines
├── notebooks/            # Exploratory analysis and model benchmarking experiments
├── src/                  # Core NLP, feature extraction, and explainability modules
├── tests/                # Unit test suite
├── Dockerfile            # Container configuration for deployment
├── requirements.txt      # Dependency specification
└── README.md             # Project documentation
```

---

## 🚀 Key Modules
1. **Linguistic Preprocessing & WhatsApp Marker Extraction**: Cleans noise while extracting forward-specific heuristic signals (urgency index, capitalization ratio, punctuation intensity).
2. **NLP Classification Engine**: TF-IDF n-gram modeling paired with calibrated linear classifiers.
3. **Explainable AI (XAI) with LIME**: Transparent token-level attribution so users understand *why* a message was flagged.
4. **Interactive Web Dashboard**: Streamlit interface allowing users to paste messages or test viral examples in real-time.
