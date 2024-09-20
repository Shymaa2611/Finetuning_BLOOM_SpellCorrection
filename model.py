from transformers import BitsAndBytesConfig
import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
from peft import LoraConfig

def model_Quantization():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) 
    
    #token="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"  
    model_name = "bigscience/bloomz-560m"
   
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    model = BloomForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config
    
    )
    return model, tokenizer
