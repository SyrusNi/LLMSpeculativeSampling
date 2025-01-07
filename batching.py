import torch
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
from sampling.utils import sample, norm_logits

MODELZOO = {
    # models I have tried
    "llama1b": "D:\Data\LLMSpeculativeSampling\model\TinyLlama-1.1B",
    "llama2-7b": "D:\Data\LLMSpeculativeSampling\model\Llama-2-7b",
    "t5-small": r"D:\Data\LLMSpeculativeSampling\model\flan-t5-small",
    "bloom-560m": r"D:\Data\LLMSpeculativeSampling\model\bloom-560m",
    "bloom7b": r"D:\Data\LLMSpeculativeSampling\model\bloomz-7b1",
    # models for reference
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="This algorithm is being created in order to")
    parser.add_argument('--model_name', type=str, default="llama2-7b")
    #parser.add_argument('--temperature', type=float, default=1.0, help='temperature for approx model')
    #parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    #parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    #parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--test_time', type=int, default=10, help='test time')
    parser.add_argument('--gamma', '-g', type=int, default=5, help='guess time')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--num_tokens', '-t', type=int, default=20, help='tokens number of input_ids')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size.')
    args = parser.parse_args()
    return args

def benchmark(fn, test_time : int = 10, *args, **kwargs):
    start = time.time()
    for _ in range(test_time):
        output = fn(*args, **kwargs)
    end = time.time()
    return (end-start) / test_time

def generate(model_name, batch_size=10, num_tokens=20, gamma_range=7, step=1, test_time=10):
    # load model and tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(model)
    print(f"begin loading models: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', load_in_4bit=True)

    #input_ids = tokenizer.encode(input_text, return_tensors='pt')
    #input_ids = input_ids.repeat(batch_size)

    # computation at initial stage
    @torch.no_grad()
    def initial_stage(input_ids):
        output = model(input_ids)
        prob = norm_logits(output.logits[:, -1, :], temperature=1, top_k=20, top_p=0.9)
        #id_next = sample(prob)
        torch.cuda.synchronize()

    # computation in the autoregressive process
    @torch.no_grad()
    def ar_stage(last_ids, past_key_values):
        output = model(last_ids, past_key_values = past_key_values, use_cache=True)
        not_cached_q = output.logits
        for i in range(not_cached_q.shape[-2]):   
            not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], temperature=1, top_k=20, top_p=0.9)
        #id_next = sample(not_cached_q[:, -1, :])
        torch.cuda.synchronize()

    gammas = np.arange(1, gamma_range, step)
    init = np.zeros_like(gammas)
    ar = np.zeros_like(gammas)
    #batchs = np.arange(batch_range)

    print(f'start test: gamma range {gamma_range}, step {step}, batch size {batch_size}, input length {num_tokens}')

    # create input_ids of shape (batch_size, num_tokens)
    input_ids = torch.randint(1, 32000, (batch_size, num_tokens), device=model.device)
    output = model(input_ids)

    for i in range(len(gammas)):
        
        gamma = gammas[i]
        last_ids = torch.randint(1, 32000, (batch_size, gamma), device=model.device)

        latency_init = benchmark(initial_stage, test_time, input_ids)
        latency_ar = benchmark(ar_stage, test_time, last_ids, output.past_key_values)

        init[i] = num_tokens*batch_size/latency_init
        ar[i] = gamma*batch_size/latency_ar

    assert latency_init > 0 and latency_ar > 0

    print(gammas[0]/ar[0])

    return gammas, init, ar

def plot_batch_inference(batchs, speed_1, speed_2):
    # plot the speed of inference under parallelism
    plt.plot(batchs, speed_1, marker = 'o', linestyle='-', label='ar')
    #plt.plot(batchs, speed_2, marker = 'o', linestyle='-', label='init')
    plt.xlabel('Gamma')
    plt.ylabel('Tokens/s')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    gammas, init, ar = generate(MODELZOO[args.model_name], args.batch_size, args.num_tokens, gamma_range=args.gamma, step=args.step)
    plot_batch_inference(gammas, ar, init)

