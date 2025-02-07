# cmd for speculative sampling
python main.py --approx_model_name models/TinyLlama_v1.1 --target_model_name models/vicuna-7b-v1.3 --approx_tmp 0.01 -b -g 7 -M 60

# cmd for batch test
python batching.py --model_name models/TinyLlama-1.1B-Chat-v1.0 -g 10 -t 20 -b 1
python batching.py --model_name models/vicuna-7b-v1.3 -g 10 -t 20 -b 1

# cmd for huggingface implementation
python hg_imp.py --approx_model_name models/llama-68m --target_model_name models/vicuna-7b-v1.3 -s 77 -b -g 7 -M 60