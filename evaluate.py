import torch
from datasets import load_metric
from transformers import BloomTokenizerFast, BloomForCausalLM
from tqdm import tqdm
from dataset import get_data


def load_model_and_tokenizer(model_path):
    model = BloomForCausalLM.from_pretrained(model_path)
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloomz-560m")
    return model, tokenizer

def spell_correct(input_text, model, tokenizer, max_len=150):
    inputs = tokenizer(input_text, return_tensors='pt', max_length=max_len, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'].to(model.device),
            max_length=max_len,
            num_return_sequences=1,
            early_stopping=True
        )
    
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def evaluate_model(model, tokenizer, test_loader):
    bleu = load_metric("sacrebleu")  
    correct_count = 0
    total_count = 0
    predictions = []
    references = []

    model.eval()
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)
        clean_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
        distorted_texts = [tokenizer.decode(input_id, skip_special_tokens=True) for input_id in input_ids]
        for distorted_text, clean_text in zip(distorted_texts, clean_texts):
            predicted_text = spell_correct(distorted_text, model, tokenizer)
            predictions.append(predicted_text)
            references.append([clean_text])  
            if predicted_text.strip() == clean_text.strip():
                correct_count += 1
            total_count += 1
    accuracy = correct_count / total_count
    bleu_score = bleu.compute(predictions=predictions, references=references)

    return accuracy, bleu_score

if __name__ == "__main__":
    model_path = "/kaggle/working/Finetuning_BLOOM_SpellCorrection/Finetuning_BLOOM_SpellCorrection/bloomspellCorrection/checkpoint-874"
    model, tokenizer = load_model_and_tokenizer(model_path)
    train_data, test_data, train_loader, test_loader=get_data('EnglishDataset/data.csv',tokenizer)
    accuracy, bleu_score = evaluate_model(model, tokenizer, test_loader)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"BLEU Score: {bleu_score['score']:.2f}")
