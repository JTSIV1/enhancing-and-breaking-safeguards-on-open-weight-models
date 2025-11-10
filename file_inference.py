import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import json
from dotenv import load_dotenv
import os

TEST_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model_map = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "gpt-oss": "openai/gpt-oss-20b",
    "deepseek": "deepseek-ai/deepseek-math-7b-instruct"
}

def load_model_and_tokenizer(model_id: str, device_map: str = "auto"):
    print(f"Loading model: {model_id} with 4-bit quantization...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    try:
        # 1. Load Tokenizer
        load_dotenv()
        access_token = os.getenv("HUGGING_FACE_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, token=access_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set tokenizer.pad_token to tokenizer.eos_token.")
            
        # 2. Load Model with Quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            token=access_token
        )
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully. Weights are 4-bit. Device map: {device_map}.")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"--- ERROR LOADING MODEL {model_id} ---")
        print("Please ensure:")
        print("1. All required packages (torch, transformers, accelerate, bitsandbytes) are installed.")
        print("2. You have a compatible NVIDIA GPU (if using 'auto' device map).")
        print("3. You have accepted the model license terms on Hugging Face (e.g., for Llama models).")
        print(f"Detailed error: {e}")
        return None, None

if __name__ == '__main__':
    model = sys.argv[1]
    input_file = sys.argv[2]
    out_file = sys.argv[3]

    model_id = model_map.get(model, TEST_MODEL_ID)

    print(f"Attempting to load a test model: {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, device_map="cpu")

    with open(input_file, "r") as f:
        prompt_data = json.load(f)
    
    if model and tokenizer:
        outputs = []
        for prompt in prompt_data:
            test_prompt = ''
            for question in prompt:
                test_prompt += "\nUser request: " + question + "\nResponse: "
            test_prompt = test_prompt.strip()
            
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            
            with torch.no_grad():
                output = model.generate(
                    input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=50, 
                    do_sample=True, 
                    temperature=0.7
                )
            
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            response = response[len(test_prompt):]
            outputs.append(response)

        with open(out_file, "w") as f:
            json.dump(outputs, f)
        print(f"Test model run successfully. Output saved to {out_file}")
    else:
        print("Failed to load test model.")