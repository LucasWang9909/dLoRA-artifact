import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_adapter(base_model_id, adapter_path, name):
    print(f"开始测试适配器: {name}")
    
    # 加载基础模型
    print("加载基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载适配器
    print("加载适配器...")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"适配器 {name} 加载成功!")
        
        # 简单的推理测试
        test_input = "predict the next token of the following sentence: I love"
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=200,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"生成文本: {generated_text}")
        return True
    except Exception as e:
        print(f"适配器 {name} 加载失败: {str(e)}")
        return False

if __name__ == "__main__":
    base_model_id = "meta-llama/Llama-2-7b-chat-hf"
    
    # 测试第一个适配器
    cat_adapter_path = "models/adapters/cat"
    cat_result = test_adapter(base_model_id, cat_adapter_path, "CAT")
    
    # 测试第二个适配器
    fin_adapter_path = "models/adapters/fin"
    fin_result = test_adapter(base_model_id, fin_adapter_path, "FinGPT")
    
    if cat_result and fin_result:
        print("两个适配器都测试成功!")
    else:
        print("适配器测试未全部成功，请检查错误信息。") 