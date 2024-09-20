import torch
from datasets import load_metric
from transformers import BloomTokenizerFast, BloomForCausalLM
from tqdm import tqdm
from dataset import get_data


def load_model_and_tokenizer(model_path):
    model = BloomForCausalLM.from_pretrained(model_path)
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloomz-560m")
    return model, tokenizer

def spell_correct(input_text, model, tokenizer, max_len=20):
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

from datasets import load_metric

def evaluate_model(model, tokenizer, test_loader):
    # Load the BLEU metric
    bleu_metric = load_metric("bleu",trust_remote_code=True)
    
    model.eval()
    total_accuracy = 0
    total_bleu_score = 0
    num_samples = 0

    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Calculate accuracy
        total_accuracy += sum(pred == label for pred, label in zip(decoded_preds, decoded_labels))
        num_samples += len(decoded_labels)

        # Update BLEU score
        bleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    # Calculate final metrics
    accuracy = total_accuracy / num_samples if num_samples > 0 else 0
    bleu_score = bleu_metric.compute()['score']  # This will give the BLEU score

    return accuracy, bleu_score

if __name__ == "__main__":
    model_path = "/kaggle/working/Finetuning_BLOOM_SpellCorrection/Finetuning_BLOOM_SpellCorrection/bloomspellCorrection/checkpoint-874"
    model, tokenizer = load_model_and_tokenizer(model_path)
    train_data, test_data, train_loader, test_loader=get_data('EnglishDataset/data.csv',tokenizer)
    accuracy, bleu_score = evaluate_model(model, tokenizer, test_loader)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"BLEU Score: {bleu_score['score']:.2f}")
