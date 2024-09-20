from transformers import BitsAndBytesConfig
import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
from peft import LoraConfig

def model_Quantization():
    model_name = "bigscience/bloomz-560m"
   
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    model = BloomForCausalLM.from_pretrained(
        model_name, 
    
    )
    return model, tokenizer
