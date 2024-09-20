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

def Lora_Configuration():
  lora_config=LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
   )
  return lora_config

