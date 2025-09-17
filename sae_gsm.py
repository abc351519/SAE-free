# 加载模型
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformer_lens import HookedTransformer
import argparse
from sae_lens import SAE
import re
parser = argparse.ArgumentParser()
    
# 添加参数
# parser.add_argument("--model",  type=str, required=True, help="")
parser.add_argument("--steering", action="store_true",help="")

parser.add_argument("--feature_idx", type=int, default=695, help="")
parser.add_argument("--strength",type=float,default=0.5)
parser.add_argument("--max_act",type=float, default=1)
parser.add_argument("--max_seq",type=int,default=2000)
parser.add_argument("--model_name",type = str, help = "d_llama3-8b, llama3-8b, gemma-9b")
args = parser.parse_args()



model_map = {
    "llama3-8b":"/lzh/model/llama-3-8b-it",
    "d_llama3-8b":"/lzh/model/distill-llama3-8b",
    "gemma-2-9b-it":"/lzh/model/gemma-2-9b-it"
}

data_map = {
    "llama3-8b":"/lzh/cot/distill/GPU_run/gsm/gsm_300_judge.jsonl",
    "d_llama3-8b":"/lzh/cot/distill/GPU_run/gsm/d_gsm_300_judge.jsonl",
    "gemma-2-9b-it":"/lzh/cot/distill/GPU_run/gsm/g_gsm_300_judge.jsonl"
}

model_path = model_map[args.model_name]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,trust_remote_code=True)
model.to("cuda")

def utils_sample(sample):
    question = sample["question"]
    answer = sample["answer"]
    prompt = f'''Answer the following question step by step in short and put the final answer in \\boxed{{}}. Question:{question}'''
    # prompt = f'''Please briefly answer the following questions step by step and put the final answer in \\boxed{{}}. Question:{question}'''
    extract_answer = answer.split('####')[-1].strip()

    return {"question":prompt, "answer":extract_answer,}


import json 

if args.steering:
    selected_questions = []
    data_dir = data_map[args.model_name]
    with open(data_dir,"r",encoding="utf8") as f:
        for line in f:
            selected_questions.append(json.loads(line))
    processed_questions = selected_questions
    # sae, cfg_dict, sparsity = SAE.from_pretrained(release = "llama_scope_r1_distill", 
    #                                           sae_id = "l15r_800m_openr1_math",
    #                                           device = "cuda")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-9b-it-res-canonical",
    sae_id = "layer_9/width_16k/canonical",
    device = "cuda"
    )
    
    
    def adjust_residual_hook():
        def hook_fn(module, input, output):
            original_dtype = output.dtype
            resid_float = output.float()
            feature_activations = sae.encode(resid_float)
            reconstructed = sae.decode(feature_activations)
            error = resid_float - reconstructed
            feature_activations[..., args.feature_idx] = args.strength * args.max_act
            resid_hat = sae.decode(feature_activations) + error
            before_norm = torch.norm(resid_float, p=2, dim=-1, keepdim=True)
            after_norm = torch.norm(resid_hat, p=2, dim=-1, keepdim=True)
            resid_hat = resid_hat * (before_norm / after_norm)
           
            # resid_hat.zero_()
            # 写回干预结果（保持原始dtype）
            resid_post = resid_hat.to(original_dtype)
            
            return resid_post
        return hook_fn
    print("add mlp hook")
    for i, layer in enumerate(model.model.layers):
        if i == 15:
            layer.mlp.register_forward_hook(adjust_residual_hook())
             
    acc = 0
    all_num = 0
    token_num = 0
    for i,ele in enumerate(processed_questions):
        prompt = ele["question"]
        answer = ele["answer"]
        inputs = tokenizer(prompt, return_tensors="pt")
        ids = inputs.input_ids.to("cuda")
        # 生成输出
    
        with torch.no_grad():
            outputs = model.generate(
            ids,
           
            max_new_tokens=args.max_seq,          # 保守长度限制
            # early_stopping=True,          # 检测EOS提前停止
            repetition_penalty=1.2,
            # temperature=0.5,
            # top_p=0.85,
            # do_sample=True,
            # num_beams=1,
            # eos_token_id=tokenizer.eos_token_id,
            # sto÷p_sequences=["###"]
            # logits_processor=[EarlyTerminationLogitsProcessor()],
        )
            # 解码输出（跳过提示词部分）
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        numbers_token = outputs[0].shape[0]-inputs.input_ids.shape[1]
        

        
        if numbers_token==args.max_seq:
            output_text = prompt + response + "Yeah, I think that's right.\n\n**Final Answer**\n"
            toks = tokenizer(output_text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=toks['input_ids'],
                    attention_mask=toks['attention_mask'],
                    max_new_tokens=20,  # Generate only the final answer
                    do_sample=True
                )
            output_text = tokenizer.decode(output_ids[0][toks.input_ids.shape[1]:],skip_special_tokens = True)

            response = response + "Yeah, I think that's right.\n\n**Final Answer**\n" + output_text
            
            numbers_token = output_ids[0].shape[0]-inputs.input_ids.shape[1]
            
        all_num+=1
        token_num+=numbers_token
        print("回答：", response)
        print("正确:", answer)
        print("numbers  token", numbers_token)
        if args.model_name.startswith("d"):
            ele["d_final_cot"] = response
            ele["d_final_tokens"] = token_num/(i+1)
            
        else:
            if args.model_name.startswith("g"):
                ele["g_final_cot"] = response
                ele["g_final_tokens"] = token_num/(i+1)
            else:
                ele["final_cot"] = response
                ele["final_tokens"] = token_num/(i+1)
    with open(f"./final_g_gsm_300_judge_{args.feature_idx}_{args.max_act}_{args.strength}.jsonl","w",encoding = "utf8") as fw:
        for ele in processed_questions:
            fw.write(json.dumps(ele,ensure_ascii=False)+"\n")        
    
    
    print("acc",acc,"all_num",all_num,"token_num",token_num/all_num)
    
if not args.steering:
    selected_questions = []
    with open('./gsm_300.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            selected_questions.append(json.loads(line))
    processed_questions = [utils_sample(ele) for ele in selected_questions]
    acc = 0
    all_num = 0
    token_num = 0
    for i,ele in enumerate(processed_questions):
        prompt = ele["question"]
        answer = ele["answer"]
        inputs = tokenizer(prompt, return_tensors="pt")
        ids = inputs.input_ids.to("cuda")
        # 生成输出
    
        with torch.no_grad():
            outputs = model.generate(
            ids,
           
            max_new_tokens=args.max_seq,          # 保守长度限制
            # early_stopping=True,          # 检测EOS提前停止
            repetition_penalty=1.2,
            # temperature=0.5,
            # top_p=0.85,
            # do_sample=True,
            # num_beams=1,
            # eos_token_id=tokenizer.eos_token_id,
            # sto÷p_sequences=["###"]
            # logits_processor=[EarlyTerminationLogitsProcessor()],
        )
            # 解码输出（跳过提示词部分）
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        numbers_token = outputs[0].shape[0]-inputs.input_ids.shape[1]
        
        
        if numbers_token==args.max_seq:
            output_text = prompt + response + "Yeah, I think that's right.\n\n**Final Answer**\n"
            toks = tokenizer(output_text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=toks['input_ids'],
                    attention_mask=toks['attention_mask'],
                    max_new_tokens=256,  # Generate only the final answer
                    do_sample=True
                )
            output_text = tokenizer.decode(output_ids[0][toks.input_ids.shape[1]:],skip_special_tokens = True)

            response = response + "Yeah, I think that's right.\n\n**Final Answer**\n" + output_text
            numbers_token = output_ids[0].shape[0]-inputs.input_ids.shape[1]
        all_num+=1
        token_num+=numbers_token
        print("回答：", response)
        print("正确:", answer)
        print("numbers  token", numbers_token)

        ele["g_orign_cot"] = response
        ele["g_orign_tokens"] = token_num/(i+1)
    with open("./g_gsm_300_judge.jsonl","w",encoding="utf8") as fw:
        for ele in processed_questions:
            fw.write(json.dumps(ele,ensure_ascii=False)+"\n")

    print("acc",acc,"all_num",all_num,"token_num",token_num/all_num)
    
    
