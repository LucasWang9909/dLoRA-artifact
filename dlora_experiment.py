#!/usr/bin/env python3
"""
dLoRA单机性能实验脚本
本脚本测试dLoRA在单节点上不同推理模式的性能差异
"""

import os
import time
import argparse
import torch
import numpy as np
import psutil
import threading
import json
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from peft import PeftModel

from vllm.config import ModelConfig, LoRaConfig, ParallelConfig, SchedulerConfig, ExecType
from vllm.worker.worker import Worker
from vllm.worker.lora_engine import LlamaLoRaEngine

# GPU监控函数
def monitor_gpu(stop_event, gpu_stats, interval=0.1):
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设使用第一个GPU
    
    while not stop_event.is_set():
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        gpu_stats.append({
            'timestamp': time.time(),
            'memory_used': info.used / 1024**2,  # MB
            'memory_total': info.total / 1024**2,  # MB
            'gpu_util': util.gpu  # %
        })
        time.sleep(interval)
    
    pynvml.nvmlShutdown()

class DLoRAExperiment:
    def __init__(self, base_model_id="meta-llama/Llama-2-7b-chat-hf"):
        """初始化实验环境"""
        self.base_model_id = base_model_id
        self.adapters = {
            "CAT": "models/adapters/cat",
            "FinGPT": "models/adapters/fin"
        }
        
        # 加载标准tokenizer用于生成测试输入
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 设置测试输入
        self.test_prompts = [
            "Explain the basic concept of artificial intelligence",
            "Analyze the global economic situation in 2023",
            "Write a simple Python function to calculate Fibonacci sequence",
            "Discuss the main impacts of climate change",
            "Recommend a science fiction novel and briefly outline its content"
        ]
        
        self.results = {
            "merged_mode": {},
            "unmerged_mode": {},
            "dynamic_mode": {}
        }
        
        # 尝试修复CAT适配器
        try:
            self.fix_cat_adapter()
        except Exception as e:
            print(f"Warning: Failed to fix CAT adapter: {str(e)}")
    
    def fix_cat_adapter(self):
        """尝试修复CAT适配器的兼容性问题"""
        import json
        import os
        
        cat_config_path = os.path.join("models/adapters/cat", "adapter_config.json")
        if os.path.exists(cat_config_path):
            with open(cat_config_path, 'r') as f:
                config = json.load(f)
            
            # 移除不兼容的loftq_config参数
            if 'loftq_config' in config:
                del config['loftq_config']
                
            # 保存修改后的配置
            with open(cat_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("CAT adapter config fixed successfully")
    
    def setup_worker(self, lora_config=None, exec_type=ExecType.LORA):
        """设置dLoRA Worker"""
        if lora_config is None:
            lora_config = LoRaConfig(max_r=8, num_models=2, gpu_capacity=2)
        
        model_config = ModelConfig(
            model=self.base_model_id,
            tokenizer=self.base_model_id,
            tokenizer_mode="auto",
            trust_remote_code=True,
            download_dir=None,
            use_np_weights=False,
            use_dummy_weights=False,
            dtype="auto",
            seed=0
        )
        
        parallel_config = ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
            worker_use_ray=False
        )
        
        scheduler_config = SchedulerConfig(
            max_num_batched_tokens=2048,
            max_num_seqs=256,
            max_model_len=4096,
            policy="fcfs"
        )
        
        worker = Worker(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            lora_config=lora_config,
            exec_type=exec_type
        )
        
        return worker
    
    def prepare_test_inputs(self, num_samples=10):
        """准备测试输入数据"""
        all_inputs = []
        for _ in range(num_samples):
            prompt = np.random.choice(self.test_prompts)
            all_inputs.append(prompt)
        return all_inputs
    
    def test_merged_mode(self, batch_sizes=[1, 4, 8, 16]):
        """测试合并模式性能 - 等待相同类型请求批处理"""
        print("\nTesting Merged Mode Performance...")
        
        # 初始化结果记录
        self.results["merged_mode"] = {
            "batch_size": batch_sizes,
            "latency": [],
            "throughput": [],
            "gpu_util": [],
            "memory_used": []
        }
        
        # 测试两种不同的适配器
        adapter_paths = {
            "FinGPT": "models/adapters/fin",
            "CAT": "models/adapters/cat"
        }
        
        for adapter_name, adapter_path in adapter_paths.items():
            print(f"\n  Testing adapter: {adapter_name}")
        
            for batch_size in batch_sizes:
                print(f"    Batch size: {batch_size}")
                
                # 准备测试数据 - 全部相同类型请求
                test_inputs = self.prepare_test_inputs(num_samples=batch_size * 3)
                
                # 监控GPU
                gpu_stats = []
                stop_event = threading.Event()
                monitor_thread = threading.Thread(
                    target=monitor_gpu, 
                    args=(stop_event, gpu_stats)
                )
                monitor_thread.start()
                
                try:
                    # 测试处理时间
                    start_time = time.time()
                    
                    # 使用标准HF接口测试
                    model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_id,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    
                    # 加载指定适配器
                    lora_model = PeftModel.from_pretrained(model, adapter_path)
                    
                    # 合并权重
                    lora_model = lora_model.merge_and_unload()
                    
                    # 批量处理
                    all_latencies = []
                    for i in range(0, len(test_inputs), batch_size):
                        batch = test_inputs[i:i+batch_size]
                        
                        # 批量编码
                        inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(lora_model.device)
                        
                        # 批量生成
                        gen_start = time.time()
                        with torch.no_grad():
                            outputs = lora_model.generate(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_new_tokens=20,
                                do_sample=True,
                                top_p=0.9,
                                temperature=0.7
                            )
                        gen_end = time.time()
                        
                        # 计算每个token的平均延迟
                        new_tokens = outputs.size(1) - inputs["input_ids"].size(1)
                        latency_per_token = (gen_end - gen_start) / new_tokens
                        all_latencies.append(latency_per_token)
                    
                    total_time = time.time() - start_time
                    avg_latency = np.mean(all_latencies) * 1000  # ms/token
                    throughput = len(test_inputs) / total_time  # 请求/秒
                    
                    # 记录结果 - 多个适配器的结果保存在单独的字段中
                    if f"latency_{adapter_name}" not in self.results["merged_mode"]:
                        self.results["merged_mode"][f"latency_{adapter_name}"] = []
                        self.results["merged_mode"][f"throughput_{adapter_name}"] = []
                        self.results["merged_mode"][f"gpu_util_{adapter_name}"] = []
                        self.results["merged_mode"][f"memory_used_{adapter_name}"] = []
                    
                    self.results["merged_mode"][f"latency_{adapter_name}"].append(avg_latency)
                    self.results["merged_mode"][f"throughput_{adapter_name}"].append(throughput)
                    
                    # 计算平均GPU利用率和内存
                    if gpu_stats:
                        avg_gpu_util = np.mean([stat["gpu_util"] for stat in gpu_stats])
                        avg_mem_used = np.mean([stat["memory_used"] for stat in gpu_stats])
                        
                        self.results["merged_mode"][f"gpu_util_{adapter_name}"].append(avg_gpu_util)
                        self.results["merged_mode"][f"memory_used_{adapter_name}"].append(avg_mem_used)
                    
                    print(f"      Average latency: {avg_latency:.2f} ms/token")
                    print(f"      Throughput: {throughput:.2f} requests/second")
                    
                finally:
                    # 停止GPU监控
                    stop_event.set()
                    monitor_thread.join()
                    
                    # 清理
                    del model
                    del lora_model
                    torch.cuda.empty_cache()
        
        # 汇总两个适配器的平均结果
        for adapter_name in adapter_paths.keys():
            if f"latency_{adapter_name}" in self.results["merged_mode"]:
                for i, batch_size in enumerate(batch_sizes):
                    if i < len(self.results["merged_mode"][f"latency_{adapter_name}"]):
                        # 添加到合并模式的总结果中
                        if i >= len(self.results["merged_mode"]["latency"]):
                            self.results["merged_mode"]["latency"].append(
                                self.results["merged_mode"][f"latency_{adapter_name}"][i]
                            )
                            self.results["merged_mode"]["throughput"].append(
                                self.results["merged_mode"][f"throughput_{adapter_name}"][i]
                            )
                            if f"gpu_util_{adapter_name}" in self.results["merged_mode"]:
                                self.results["merged_mode"]["gpu_util"].append(
                                    self.results["merged_mode"][f"gpu_util_{adapter_name}"][i]
                                )
                                self.results["merged_mode"]["memory_used"].append(
                                    self.results["merged_mode"][f"memory_used_{adapter_name}"][i]
                                )
                        else:
                            self.results["merged_mode"]["latency"][i] += self.results["merged_mode"][f"latency_{adapter_name}"][i]
                            self.results["merged_mode"]["throughput"][i] += self.results["merged_mode"][f"throughput_{adapter_name}"][i]
                            if f"gpu_util_{adapter_name}" in self.results["merged_mode"]:
                                self.results["merged_mode"]["gpu_util"][i] += self.results["merged_mode"][f"gpu_util_{adapter_name}"][i]
                                self.results["merged_mode"]["memory_used"][i] += self.results["merged_mode"][f"memory_used_{adapter_name}"][i]
        
        # 计算平均值
        adapter_count = len(adapter_paths)
        for i in range(len(self.results["merged_mode"]["latency"])):
            self.results["merged_mode"]["latency"][i] /= adapter_count
            self.results["merged_mode"]["throughput"][i] /= adapter_count
            if i < len(self.results["merged_mode"]["gpu_util"]):
                self.results["merged_mode"]["gpu_util"][i] /= adapter_count
                self.results["merged_mode"]["memory_used"][i] /= adapter_count
    
    def test_unmerged_mode(self, batch_sizes=[1, 4, 8, 16]):
        """测试非合并模式性能"""
        print("\nTesting Unmerged Mode Performance...")
        
        # 初始化结果记录
        self.results["unmerged_mode"] = {
            "batch_size": batch_sizes,
            "latency": [],
            "throughput": [],
            "gpu_util": [],
            "memory_used": []
        }
        
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")
            
            # 准备测试数据
            test_inputs = self.prepare_test_inputs(num_samples=batch_size * 3)
            
            # 监控GPU
            gpu_stats = []
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=monitor_gpu, 
                args=(stop_event, gpu_stats)
            )
            monitor_thread.start()
            
            try:
                # 测试处理时间
                start_time = time.time()
                
                # 使用标准HF接口测试
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # 加载FinGPT适配器（已知可用）
                fin_adapter_path = "models/adapters/fin"
                lora_model = PeftModel.from_pretrained(model, fin_adapter_path)
                
                # 批量处理
                all_latencies = []
                for i in range(0, len(test_inputs), batch_size):
                    batch = test_inputs[i:i+batch_size]
                    
                    # 批量编码
                    inputs = self.tokenizer(batch, return_tensors="pt", padding=True).to(lora_model.device)
                    
                    # 批量生成
                    gen_start = time.time()
                    with torch.no_grad():
                        outputs = lora_model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=20,
                            do_sample=True,
                            top_p=0.9,
                            temperature=0.7
                        )
                    gen_end = time.time()
                    
                    # 计算每个token的平均延迟
                    new_tokens = outputs.size(1) - inputs["input_ids"].size(1)
                    latency_per_token = (gen_end - gen_start) / new_tokens
                    all_latencies.append(latency_per_token)
                
                total_time = time.time() - start_time
                avg_latency = np.mean(all_latencies) * 1000  # ms/token
                throughput = len(test_inputs) / total_time  # 请求/秒
                
                # 记录结果
                self.results["unmerged_mode"]["latency"].append(avg_latency)
                self.results["unmerged_mode"]["throughput"].append(throughput)
                
                # 计算平均GPU利用率和内存
                if gpu_stats:
                    avg_gpu_util = np.mean([stat["gpu_util"] for stat in gpu_stats])
                    avg_mem_used = np.mean([stat["memory_used"] for stat in gpu_stats])
                    self.results["unmerged_mode"]["gpu_util"].append(avg_gpu_util)
                    self.results["unmerged_mode"]["memory_used"].append(avg_mem_used)
                
                print(f"    Average latency: {avg_latency:.2f} ms/token")
                print(f"    Throughput: {throughput:.2f} requests/second")
                
            finally:
                # 停止GPU监控
                stop_event.set()
                monitor_thread.join()
                
                # 清理
                del model
                del lora_model
                torch.cuda.empty_cache()
    
    def test_dynamic_mode(self, request_distributions=[0.5, 0.7, 0.9]):
        """测试动态批处理策略 - 根据请求分布动态决定使用哪种模式
        
        Args:
            request_distributions: 主要适配器的请求占比列表
        """
        print("\nTesting Dynamic Batching Strategy...")
        
        # 初始化结果记录
        self.results["dynamic_mode"] = {
            "distribution": request_distributions,
            "latency": [],
            "throughput": [],
            "gpu_util": [],
            "memory_used": []
        }
        
        batch_size = 16  # 固定批处理大小
        
        for distribution in request_distributions:
            print(f"  Request distribution: {distribution*100:.0f}% concentrated on primary adapter")
            
            # 准备混合测试数据
            num_samples = batch_size * 3
            test_inputs = []
            adapter_types = []  # 0: FinGPT, 1: CAT
            
            for _ in range(num_samples):
                prompt = np.random.choice(self.test_prompts)
                test_inputs.append(prompt)
                
                # 根据分布确定使用哪个适配器
                if np.random.random() < distribution:
                    adapter_types.append(0)  # 主要适配器 (FinGPT)
                else:
                    adapter_types.append(1)  # 次要适配器 (CAT)
            
            # 监控GPU
            gpu_stats = []
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=monitor_gpu, 
                args=(stop_event, gpu_stats)
            )
            monitor_thread.start()
            
            try:
                # 测试处理时间
                start_time = time.time()
                
                # 加载基础模型
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # 确定主要和次要适配器
                adapter_counts = {0: adapter_types.count(0), 1: adapter_types.count(1)}
                primary_adapter_type = 0 if adapter_counts[0] >= adapter_counts[1] else 1
                secondary_adapter_type = 1 if primary_adapter_type == 0 else 0
                
                primary_adapter_path = "models/adapters/fin" if primary_adapter_type == 0 else "models/adapters/cat"
                secondary_adapter_path = "models/adapters/cat" if primary_adapter_type == 0 else "models/adapters/fin"
                
                primary_adapter_name = "FinGPT" if primary_adapter_type == 0 else "CAT"
                secondary_adapter_name = "CAT" if primary_adapter_type == 0 else "FinGPT"
                
                print(f"    Primary adapter: {primary_adapter_name} ({adapter_counts[primary_adapter_type]} requests)")
                print(f"    Secondary adapter: {secondary_adapter_name} ({adapter_counts[secondary_adapter_type]} requests)")
                
                # dLoRA核心策略: 
                # 1. 当请求高度集中于一个adapter时，使用合并模式
                # 2. 当请求分散于多个adapter时，使用非合并模式实现跨adapter批处理
                if distribution >= 0.7:  # 请求高度集中
                    print("    Using merged mode (high concentration)")
                    
                    # 合并主要适配器
                    primary_model = PeftModel.from_pretrained(base_model, primary_adapter_path)
                    primary_model = primary_model.merge_and_unload()
                    
                    # 为次要适配器类型加载单独的模型实例
                    secondary_model = None
                    if adapter_counts[secondary_adapter_type] > 0:
                        secondary_model = PeftModel.from_pretrained(
                            AutoModelForCausalLM.from_pretrained(
                                self.base_model_id,
                                torch_dtype=torch.float16,
                                device_map="auto"
                            ), 
                            secondary_adapter_path
                        )
                else:
                    print("    Using unmerged mode (diverse requests)")
                    
                    # 加载两个未合并的适配器
                    primary_model = PeftModel.from_pretrained(base_model, primary_adapter_path)
                    secondary_model = PeftModel.from_pretrained(
                        AutoModelForCausalLM.from_pretrained(
                            self.base_model_id,
                            torch_dtype=torch.float16,
                            device_map="auto"
                        ), 
                        secondary_adapter_path
                    )
                
                # 批量处理
                all_latencies = []
                for i in range(0, len(test_inputs), batch_size):
                    batch_indices = list(range(i, min(i+batch_size, len(test_inputs))))
                    batch = [test_inputs[j] for j in batch_indices]
                    batch_adapter_types = [adapter_types[j] for j in batch_indices]
                    
                    # 对每种adapter类型分别处理请求
                    results = {}
                    gen_start = time.time()
                    
                    # 按adapter类型分组请求
                    adapter_batches = {}
                    for j, adapter_type in enumerate(batch_adapter_types):
                        if adapter_type not in adapter_batches:
                            adapter_batches[adapter_type] = []
                        adapter_batches[adapter_type].append((j, batch[j]))
                    
                    # 为每种adapter类型处理请求
                    total_new_tokens = 0
                    for adapter_type, adapter_batch in adapter_batches.items():
                        if not adapter_batch:
                            continue
                            
                        indices = [item[0] for item in adapter_batch]
                        prompts = [item[1] for item in adapter_batch]
                        
                        # 批量编码
                        model_to_use = primary_model if adapter_type == primary_adapter_type else secondary_model
                        if model_to_use is None:
                            continue
                            
                        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(model_to_use.device)
                        
                        # 批量生成
                        with torch.no_grad():
                            outputs = model_to_use.generate(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_new_tokens=20,
                                do_sample=True,
                                top_p=0.9,
                                temperature=0.7
                            )
                        
                        # 计算新生成的token数量
                        new_tokens = outputs.size(1) - inputs["input_ids"].size(1)
                        total_new_tokens += new_tokens
                        
                        # 存储每个请求的输出结果 
                        for k, idx in enumerate(indices):
                            results[idx] = outputs[k]
                    
                    gen_end = time.time()
                    
                    # 计算每个token的平均延迟
                    if total_new_tokens > 0:
                        latency_per_token = (gen_end - gen_start) / total_new_tokens
                        all_latencies.append(latency_per_token)
                
                total_time = time.time() - start_time
                avg_latency = np.mean(all_latencies) * 1000 if all_latencies else 0  # ms/token
                throughput = len(test_inputs) / total_time  # 请求/秒
                
                # 记录结果
                self.results["dynamic_mode"]["latency"].append(avg_latency)
                self.results["dynamic_mode"]["throughput"].append(throughput)
                
                # 计算平均GPU利用率和内存
                if gpu_stats:
                    avg_gpu_util = np.mean([stat["gpu_util"] for stat in gpu_stats])
                    avg_mem_used = np.mean([stat["memory_used"] for stat in gpu_stats])
                    self.results["dynamic_mode"]["gpu_util"].append(avg_gpu_util)
                    self.results["dynamic_mode"]["memory_used"].append(avg_mem_used)
                
                print(f"    Average latency: {avg_latency:.2f} ms/token")
                print(f"    Throughput: {throughput:.2f} requests/second")
                
            finally:
                # 停止GPU监控
                stop_event.set()
                monitor_thread.join()
                
                # 清理
                del base_model
                if 'primary_model' in locals():
                    del primary_model
                if 'secondary_model' in locals() and secondary_model is not None:
                    del secondary_model
                torch.cuda.empty_cache()
    
    def plot_results(self):
        """绘制实验结果图表"""
        print("\nGenerating result charts...")
        
        # 创建结果目录
        os.makedirs("results", exist_ok=True)
        
        # 保存结果数据
        with open("results/experiment_data.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # 绘制批处理大小与延迟关系图
        plt.figure(figsize=(12, 8))
        
        # 检查是否有合并模式和非合并模式的数据
        has_merged = "latency" in self.results["merged_mode"] and len(self.results["merged_mode"]["latency"]) > 0
        has_unmerged = "latency" in self.results["unmerged_mode"] and len(self.results["unmerged_mode"]["latency"]) > 0
        
        # 合并模式
        if has_merged:
            plt.subplot(2, 2, 1)
            plt.plot(
                self.results["merged_mode"]["batch_size"],
                self.results["merged_mode"]["latency"],
                'o-',
                label="Merged Mode"
            )
            if has_unmerged:
                plt.plot(
                    self.results["unmerged_mode"]["batch_size"],
                    self.results["unmerged_mode"]["latency"],
                    's-',
                    label="Unmerged Mode"
                )
            plt.xlabel("Batch Size")
            plt.ylabel("Latency (ms/token)")
            plt.title("Batch Size vs Latency")
            plt.legend()
            plt.grid(True)
            
            # 批处理大小与吞吐量关系图
            plt.subplot(2, 2, 2)
            plt.plot(
                self.results["merged_mode"]["batch_size"],
                self.results["merged_mode"]["throughput"],
                'o-',
                label="Merged Mode"
            )
            if has_unmerged:
                plt.plot(
                    self.results["unmerged_mode"]["batch_size"],
                    self.results["unmerged_mode"]["throughput"],
                    's-',
                    label="Unmerged Mode"
                )
            plt.xlabel("Batch Size")
            plt.ylabel("Throughput (requests/second)")
            plt.title("Batch Size vs Throughput")
            plt.legend()
            plt.grid(True)
        # 只有非合并模式
        elif has_unmerged:
            plt.subplot(2, 2, 1)
            plt.plot(
                self.results["unmerged_mode"]["batch_size"],
                self.results["unmerged_mode"]["latency"],
                's-',
                label="Unmerged Mode"
            )
            plt.xlabel("Batch Size")
            plt.ylabel("Latency (ms/token)")
            plt.title("Batch Size vs Latency")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(
                self.results["unmerged_mode"]["batch_size"],
                self.results["unmerged_mode"]["throughput"],
                's-',
                label="Unmerged Mode"
            )
            plt.xlabel("Batch Size")
            plt.ylabel("Throughput (requests/second)")
            plt.title("Batch Size vs Throughput")
            plt.legend()
            plt.grid(True)
        
        # 动态模式
        has_dynamic = "latency" in self.results["dynamic_mode"] and len(self.results["dynamic_mode"]["latency"]) > 0
        if has_dynamic:
            plt.subplot(2, 2, 3)
            plt.plot(
                [f"{d*100:.0f}%" for d in self.results["dynamic_mode"]["distribution"]],
                self.results["dynamic_mode"]["latency"],
                'd-',
                label="Dynamic Mode"
            )
            plt.xlabel("Request Concentration")
            plt.ylabel("Latency (ms/token)")
            plt.title("Request Distribution vs Latency")
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.plot(
                [f"{d*100:.0f}%" for d in self.results["dynamic_mode"]["distribution"]],
                self.results["dynamic_mode"]["throughput"],
                'd-',
                label="Dynamic Mode"
            )
            plt.xlabel("Request Concentration")
            plt.ylabel("Throughput (requests/second)")
            plt.title("Request Distribution vs Throughput")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("results/performance_comparison.png")
        print("  Charts saved to results/performance_comparison.png")
        
        # 绘制GPU利用率和内存使用图
        plt.figure(figsize=(12, 6))
        
        # GPU利用率
        has_merged_gpu = has_merged and "gpu_util" in self.results["merged_mode"]
        has_unmerged_gpu = has_unmerged and "gpu_util" in self.results["unmerged_mode"]
        
        if has_merged_gpu or has_unmerged_gpu:
            plt.subplot(1, 2, 1)
            if has_merged_gpu:
                plt.plot(
                    self.results["merged_mode"]["batch_size"],
                    self.results["merged_mode"]["gpu_util"],
                    'o-',
                    label="Merged Mode"
                )
            if has_unmerged_gpu:
                plt.plot(
                    self.results["unmerged_mode"]["batch_size"],
                    self.results["unmerged_mode"]["gpu_util"],
                    's-',
                    label="Unmerged Mode"
                )
            has_dynamic_gpu = has_dynamic and "gpu_util" in self.results["dynamic_mode"]
            if has_dynamic_gpu:
                plt.plot(
                    [i for i in range(len(self.results["dynamic_mode"]["gpu_util"]))],
                    self.results["dynamic_mode"]["gpu_util"],
                    'd-',
                    label="Dynamic Mode"
                )
            plt.xlabel("Batch Size/Distribution Index")
            plt.ylabel("GPU Utilization (%)")
            plt.title("GPU Utilization Comparison")
            plt.legend()
            plt.grid(True)
            
            # 内存使用
            plt.subplot(1, 2, 2)
            if has_merged_gpu:
                plt.plot(
                    self.results["merged_mode"]["batch_size"],
                    self.results["merged_mode"]["memory_used"],
                    'o-',
                    label="Merged Mode"
                )
            if has_unmerged_gpu:
                plt.plot(
                    self.results["unmerged_mode"]["batch_size"],
                    self.results["unmerged_mode"]["memory_used"],
                    's-',
                    label="Unmerged Mode"
                )
            if has_dynamic_gpu:
                plt.plot(
                    [i for i in range(len(self.results["dynamic_mode"]["memory_used"]))],
                    self.results["dynamic_mode"]["memory_used"],
                    'd-',
                    label="Dynamic Mode"
                )
            plt.xlabel("Batch Size/Distribution Index")
            plt.ylabel("GPU Memory Usage (MB)")
            plt.title("GPU Memory Usage Comparison")
            plt.legend()
            plt.grid(True)
        
            plt.tight_layout()
            plt.savefig("results/resource_usage.png")
            print("  Charts saved to results/resource_usage.png")

def main():
    parser = argparse.ArgumentParser(description="dLoRA Single-node Performance Experiment")
    parser.add_argument("--mode", type=str, default="all", 
                       choices=["all", "merged", "unmerged", "dynamic"],
                       help="Specify which experiment mode to run")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16",
                       help="Batch sizes list, comma separated")
    parser.add_argument("--distributions", type=str, default="0.5,0.7,0.9",
                       help="Request distribution list, comma separated")
    args = parser.parse_args()
    
    # 解析参数
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    distributions = [float(x) for x in args.distributions.split(",")]
    
    # 创建实验对象
    experiment = DLoRAExperiment()
    
    # 根据指定模式运行实验
    if args.mode in ["all", "merged"]:
        experiment.test_merged_mode(batch_sizes=batch_sizes)
    
    if args.mode in ["all", "unmerged"]:
        experiment.test_unmerged_mode(batch_sizes=batch_sizes)
    
    if args.mode in ["all", "dynamic"]:
        experiment.test_dynamic_mode(request_distributions=distributions)
    
    # 绘制结果
    experiment.plot_results()
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main() 