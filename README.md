# Fine-Tuning BLOOM for Spell Correction Task

![Model](media/model.png)

## Model Details
BLOOM is an autoregressive Large Language Model (LLM), trained to continue text from a prompt on vast amounts of text data using industrial-scale computational resources. As such, it is able to output coherent text in 46 languages and 13 programming languages that is hardly distinguishable from text written by humans. BLOOM can also be instructed to perform text tasks it hasn't been explicitly trained for, by casting them as text generation tasks.

## Table of Contents
    - Overview 
    - Requirements
    - Dataset
    - Installation
    - Training
    - Evaluation
    - Usage
    - Checkpoint

## Overview

This project focuses on fine-tuning the BLOOM model for the task of spell correction. The goal 
is to enhance the model's ability to correct spelling errors in input text, making it suitable for applications in text processing, chatbots, and other NLP tasks.
 

## Requirements

    Python 3.7 or higher
    PyTorch
    Transformers library
    Pandas
    Scikit-learn
    Datasets
    tqdm
You can install the required packages using pip:
``` bash
!pip install -r requirements.txt
```

## Dataset

The model is fine-tuned on a custom dataset containing pairs of distorted (incorrect) and clean (correct) word. The dataset should be in CSV format with the following columns:
    clean: The correct word.
    distorted: The word with spelling errors.

## Installation

Clone the repository and navigate into the project directory:
```bash
git clone https://github.com/Shymaa2611/Finetuning_BLOOM_SpellCorrection.git
cd Finetuning_BLOOM_SpellCorrection 
```

## Training

To fine-tune the BLOOM model, run the following command:
``` bash 
!python main.py

```

## Evaluation Metrics:

    Accuracy: The percentage of correctly predicted spellings.
    BLEU Score: Measures the similarity between the predicted and ground truth text.

## Usage

``` python

from inference import load_model_and_tokenizer,spell_correct
model_path = "checkpoint"
model, tokenizer = load_model_and_tokenizer(model_path)
input_word = "rea"
corrected_word = spell_correct(input_word, model, tokenizer)
print(f"Original: {input_sentence}")
print(f"Corrected: {corrected_sentence}")

```


## Checkpoint
can download from : https://drive.google.com/drive/folders/1WMYzNsIg0OIZqNERlw2VRifjZAF7QJhw?usp=drive_link