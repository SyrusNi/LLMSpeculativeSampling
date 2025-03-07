import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM # for T5 series model

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2
from globals import Decoder

MODELZOO = {
    # models I have tried
    'llama-68m': "D:\Data\LLMSpeculativeSampling\model\llama-68m",
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
    parser.add_argument('--approx_model_name', type=str, default="llama2-7b")
    parser.add_argument('--target_model_name', type=str, default="llama2-70b")
    parser.add_argument('--approx_tmp', type=float, default=1.0, help='temperature for approx model')
    parser.add_argument('--target_tmp', type=float, default=1.0, help='temperature for target model')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    #parser.add_argument("--load_in_4bit", type=bool, default=True, help='load in 4bit at a local computer')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    parser.add_argument('--test_time', '-t', type=int, default=10, help='number of measurements of the benchmark')
    args = parser.parse_args()
    return args


def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def benchmark(fn, print_prefix, test_time, use_profiler=True, *args, **kwargs):
    '''
    repeat the fn for [TEST_TIME] and test the average time cost
    '''
    TEST_TIME = test_time
    assert TEST_TIME > 0

    profile_filename = f"./profile_logs/{print_prefix}"
    
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / (t.elapsed / TEST_TIME)}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")

def generate(input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4, approx_tmp = 1.0, target_tmp = 1.0,
             random_seed = None, verbose = False, use_benchmark = False, test_time = 10, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       #load_in_4bit=True,
                                                       trust_remote_code=True)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       device_map="auto",
                                                       #load_in_4bit=True,
                                                       trust_remote_code=True)
    print("finish loading models")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, large_model, num_tokens, temperature = approx_tmp, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_large", test_time, use_profiling,
                  input_ids, large_model, num_tokens, temperature = approx_tmp, top_k = top_k, top_p=top_p)

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, small_model, num_tokens, temperature = target_tmp, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_small", test_time, use_profiling,
                  input_ids, small_model, num_tokens, temperature = target_tmp, top_k = top_k, top_p=top_p)
    
    '''
    torch.manual_seed(123)
    output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"deepmind's speculative_sampling: {generated_text}")  
    '''
    
    torch.manual_seed(123)
    output = speculative_sampling(input_ids, small_model, large_model, num_tokens, gamma = gamma, approx_tmp = approx_tmp, target_tmp = target_tmp,
                                  top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"google's speculative_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(speculative_sampling, "SP", test_time, use_profiling,
                  input_ids, small_model, large_model, max_len = num_tokens, gamma = gamma, approx_tmp = approx_tmp, target_tmp = target_tmp,
                  top_k = top_k, top_p=top_p, random_seed = random_seed, benchmark = True)

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.approx_model_name in MODELZOO:
        args.approx_model_name = MODELZOO[args.approx_model_name]
    if args.target_model_name in MODELZOO:
        args.target_model_name = MODELZOO[args.target_model_name]

    generate(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
             approx_tmp = args.approx_tmp, target_tmp = args.target_tmp,
             random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, test_time=args.test_time)
