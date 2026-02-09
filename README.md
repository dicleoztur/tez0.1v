# Detecting Subjectivity in Turkish News Texts  
*M.Sc. Thesis (Boğaziçi University, 2014)*

This repository contains the codebase, experimental setup, and documentation for my Master's thesis with the title Detecting Subjectivity in the News Texts in Turkish Language.

---

## Motivation and Problem Setting

Identifying subjective language is a fundamental challenge in natural language
understanding. While humans can effortlessly distinguish between factual reporting
and subjective judgement, enabling computers to make the same distinction remains
non-trivial.

Consider the difference between the statements *“The weather is rainy”* and
*“The weather is nice.”*  
The former reports an observable fact, while the latter reflects an individual
attitude that may vary across speakers. Subjectivity, in this sense, is tightly
connected to perspective, evaluation, and implicit stance.

This thesis investigates whether such distinctions can be **learned automatically**
by computational systems, focusing on **Turkish news texts**, a language and domain
that pose additional challenges due to rich morphology and limited annotated
resources.

---

## Beyond Sentiment Analysis

Although often discussed together, **subjectivity detection** and **sentiment
analysis** address distinct problems.

- Sentiment analysis primarily targets **polarity** (positive, negative, neutral).
- Subjectivity detection focuses on separating **factual voice** from **subjective
tone**, regardless of polarity.

This work positions subjectivity detection as a foundational task, with potential
impact on downstream applications such as information extraction, discourse analysis,
and word-sense disambiguation, and on understanding language and linguistic features.

---

## Methodology and Approach

The thesis proposes a supervised learning framework for subjectivity detection
based on classical machine learning techniques, emphasizing interpretability and
feature-level analysis.

The components are:

- **Features:**  
  Extraction of subjectivity cues from lexical features, emotion-laden keywords, n-grams, and
  part-of-speech (POS) tag attributes, tailored to Turkish morphology.

- **Dataset:**  
  Construction of an original dataset of Turkish news texts, labelled using a
  bespoke annotation scheme developed as part of this work.

- **Experiments:**  
  Systematic evaluation of multiple feature sets and classifier models
  (e.g. SVM, Naive Bayes) to analyze performance trade-offs and learning behaviour.

---

## Contributions

This thesis makes several contributions to the study of subjectivity detection in
Turkish:

- One of the early supervised frameworks for subjectivity detection in Turkish news
  texts
- A newly annotated corpus with a clearly defined subjectivity annotation scheme
- A feature-driven approach highlighting linguistic and morpho-syntactic cues
- Empirical insights into model and feature interactions in subjectivity learning

---

## Repository Contents

This repository includes:

- **Feature Extraction and Classification Code** 
- **Morphological Analysis Tools** for Turkish
- **Experimental Framework** supporting feature ablation and model comparison
- **Documentation** reflecting the analytical and experimental process

---

## Thesis Text

The thesis manuscript is available:

- As a PDF in this repository:  
  [msc-thesis_DicleOzturk.pdf](msc-thesis_DicleOzturk.pdf)
- On Academia.edu:  
  https://www.academia.edu/8561858/Detecting_Subjectivity_in_the_News_Texts_in_Turkish_Language_-_MS_Thesis_2014_

---

## Author


Dicle Öztürk 
dicle@lucitext.io

