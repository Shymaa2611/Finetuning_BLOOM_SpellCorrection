import torch
from transformers import BloomTokenizerFast, BloomForCausalLM

def load_model_and_tokenizer(model_path):
    model = BloomForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def spell_correct(input_text, model, tokenizer, max_len=150):
    inputs = tokenizer(input_text, return_tensors='pt', max_length=max_len, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_len,
            num_return_sequences=1,
            early_stopping=True
        )
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

if __name__ == "__main__":
    model_path = "/kaggle/working/Finetuning_BLOOM_SpellCorrection/Finetuning_BLOOM_SpellCorrection/bloomspellCorrection/checkpoint-874"
    model, tokenizer = load_model_and_tokenizer(model_path)
    input_sentence = "lfose_off"
    corrected_sentence = spell_correct(input_sentence, model, tokenizer)

    print(f"Original: {input_sentence}")
    print(f"Corrected: {corrected_sentence}")
