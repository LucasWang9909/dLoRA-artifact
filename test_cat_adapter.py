import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)

print("Loading base model...")
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", 
        torch_dtype=torch.float16,
        device_map="cpu"  # Use CPU to avoid OOM
    )
    print("Base model loaded successfully!")
except Exception as e:
    print(f"Error loading base model: {str(e)}")
    sys.exit(1)

print("Loading CAT adapter...")
try:
    cat_adapter_path = "models/adapters/cat"
    cat_model = PeftModel.from_pretrained(base_model, cat_adapter_path)
    print("CAT adapter loaded successfully!")
    
    # Try to merge the adapter
    print("Testing adapter merge...")
    merged_model = cat_model.merge_and_unload()
    print("Adapter merged successfully!")
    
except Exception as e:
    print(f"Error loading CAT adapter: {str(e)}")
    
print("Testing FinGPT adapter...")
try:
    fin_adapter_path = "models/adapters/fin"
    fin_model = PeftModel.from_pretrained(base_model, fin_adapter_path)
    print("FinGPT adapter loaded successfully!")
    
    # Try to merge the adapter
    print("Testing FinGPT adapter merge...")
    merged_model = fin_model.merge_and_unload()
    print("FinGPT adapter merged successfully!")
    
except Exception as e:
    print(f"Error loading FinGPT adapter: {str(e)}") 