from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch

model_name = r"D:\Data\LLMSpeculativeSampling\model\bloomz-7b1"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = 'left') # 默认填充方向是右侧，在推理场景下不好用
if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token # Most LLMs don't have a pad token by default
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:0', load_in_4bit=True) # 正常情况一张卡是放不下的
#model = AutoModelForSeq2SeqLM.from_pretrained(Tiny_llama_1B, device_map='cuda:0', load_in_4bit=True) # 正常情况一张卡是放不下的

tokens = tokenizer(['Hello', 'quantum computation'], return_tensors='pt', padding=True, truncation=True, max_length=15).to(model.device)
# 输入超过一个的话就需要 padding(填充), padding 需要设置 pad_token 以及 max_length; 如果输入太长也可以设置 truncation(截断)
# 直接用 tokenizer 得到的 token 是一个字典, 有'input_ids'和'attention_mask'参数
generate_ids = model.generate(**tokens, max_new_tokens=50, do_sample=True) # 注意和 model(**tokens) 作区分

ans = tokenizer.batch_decode(generate_ids, skip_special_tokens=True) # 返回一个 list
print(ans)