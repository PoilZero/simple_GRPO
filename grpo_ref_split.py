## for running
from huggingface_hub import login
login("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")
model_path = "Qwen/Qwen2.5-0.5B"

## raw
'''
    generate_mode: 首先通过分布式torch.distributed来合成大量数据并根据设计的reward对answer进行标注
    GRPO_step: 然后通过GRPO计算loss并backward
    循环往复完成训练
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, requests, io, sys
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
os.environ['TOKENIZERS_PARALLELISM'] = 'true' # 启动tokenizer并行化

beta = 0.03 # GRPO 中的 KL 散度权重参数
num_pre_Q = 8 # 每个问题生成的候选答案数量
Q_batch_size = 1  # 每批次的问题数量
all_steps = 1000
max_prompt_length = 400
save_steps = 200

ds_config = {
    "train_micro_batch_size_per_gpu": Q_batch_size*num_pre_Q,  # 每个 GPU 的微批量大小
    "gradient_accumulation_steps": 2,  # 梯度累积步数
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 1, # 使用 ZeRO stage 1
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"} # 空闲自动加载优化器到cpu
    }
}

ref_server = "http://localhost:59875"
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

def get_batch():
    '''
        从ref_server请求一个batch的数据
    '''
    try:
        # http.get获取数据集的一个batch (ref_server)
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    return data

'''
    在线加载tokenizer model=gen_model dataset
'''
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
        torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
gen_model = model

from datasets import load_dataset
#dataset = load_dataset("meta-math/GSM8K_zh", "default", split="train")
dataset = load_dataset("openai/gsm8k", "main", split="train")
# QAs map dataset to {'Q': XX, 'A':XX}
QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]

'''
    config
'''
from transformers import GenerationConfig
generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True, temperature=0.9, 
            num_return_sequences=num_pre_Q,
            pad_token_id=tokenizer.pad_token_id,
        )

system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

# 根据prompt调用tokenizer和gen_model生成answer
def gen_answers(prompts):
    tip_text = []
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_length = tip_inputs["input_ids"].shape[-1]
    if prompt_length > max_prompt_length: return []
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}
    with torch.inference_mode():
        tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    completion_ids = tip_completion_ids[:, prompt_length:]
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids]
    return answers


# 计算正确性reward：验证答案是否正确
################################## TO DO
from math_verify import parse, verify, ExprExtractionConfig
def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) # 使用正则表达式在answer中查找所有数字
    if len(nums) == 0: return -1.0
    lastnum = nums[-1] # 用answer中最后一个数字和ground_truth做比较
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1
# 计算格式reward：验证答案是否正确
################################## TO DO
def reward_format(item, answer):
    # pattern = r"^<think>(?:(?!</?think>)[\s\S]*?)</think>\s*<answer>(?:(?!</?answer>)[\s\S]*?)</answer><\|im_end\|>$"
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1

# 根据输入（采样过的QA）来调用gen_answer生成answer，并计算answer的reward
def gen_samples(inputs):
    prompts = [x["Q"] for x in inputs]
    answers = gen_answers(prompts)
    if len(answers) == 0: return None, None, None, None
    rewards = []
    for i, inp in enumerate(inputs):
        for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
            rewards.append(reward_correct(inp, a) + reward_format(inp, a)) # 奖励为正确性奖励和格式奖励之和
    prompts_text = [tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
    prompt_inputs = tokenizer(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    output_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)
    return prompt_inputs["input_ids"], output_ids["input_ids"], torch.tensor(rewards, dtype=torch.float32), answers

'''
    forward推理并生成合成高质量的answer数据（根据reward筛选）
    
    FORWARD for random num times
        each times random batch_size x 1
    and POST prompt_ids + answer_ids
'''
def generate_mode(num=10, rank=0):
    # torch.distributed.get_rank()分布式的ID（多机），一般0为首个进程（首机，主进程）
    # 仅在主进程上打印loss等信息
    if rank == 0: print('enter generate mode')
    # evaluation for num count
    for ii in range(num):
        # 随机从数据集中抽样Q_batch_size个样本
        inputs = random.sample(QAs, Q_batch_size)
        # gen_samples生成answer并评估reward
        prompt_inputs, output_ids, rewards, answers = gen_samples(inputs)
        if prompt_inputs is None: continue
        if rank == 0: 
            print('rewards:', rewards)
            if ii == 5:
                print('answers:', answers[0])
        if (rewards.max() - rewards.min()).item() < 0.01: continue
        rep = output_ids.shape[0] // prompt_inputs.shape[0]
        prompt_length = prompt_inputs.shape[1]
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        # merged_ids = prompt_ids + answer_ids
        merged_ids = torch.cat([Qrep, output_ids], dim=1)
        # stored
        xdata = make_bytes_list([json.dumps({"plen": prompt_length}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(rewards)]) 
        requests.post(f"{ref_server}/upload", data=xdata)
    if rank == 0: print('exit generate mode')

if 'genonly' in sys.argv:
    model.to('cuda')
    generate_mode(999999)
    sys.exit()

# gen_model = deepseek(gen_model)
import deepspeed
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                               model_parameters=model.parameters())
gen_model = engine

'''
    GRPO_step
    batch = {
        'plen': [batch_size, prompt_length],
        'inputs': [batch_size, inputs],
        'reward': [batch_size, reward],
    }
    return loss (for backward)
'''
def GRPO_step(batch):
    # 从gen_model合成的数据中提取数据（包含prompt answer reward）
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    rewards = batch['rewards'].to(engine.device)

    '''
        logits == answer_logits
        input_ids == prompt
    '''
    def get_per_token_logps(logits, input_ids):
        # 调整logits和input_ids的维度
        logits = logits[:, :-1, :]   # 移除最后一个logit（无对应token）
        input_ids = input_ids[:, 1:] # 移除第一个token（无对应logit）
        # 逐个样本计算对数概率（减少显存占用）
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    # 计算当前策略和参考策略的log概率
    per_token_logps = get_per_token_logps(engine(inputs).logits, inputs)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    # 计算KL散度（策略约束项）
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    # 生成掩码（忽略padding部分）
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    # 标准化奖励（计算优势函数）
    mean_grouped_rewards = rewards.view(-1, num_pre_Q).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, num_pre_Q).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_pre_Q, dim=0)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    # 计算最终损失（含策略梯度项和KL约束per_token_kl）
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss
'''
    训练循环
'''
generate_mode(rank=torch.distributed.get_rank())

from tqdm import tqdm
progress = range(1, all_steps+1)
if torch.distributed.get_rank() == 0: progress = tqdm(progress)
for step in progress:
    # forward推理并生成合成高质量的answer数据（根据reward筛选）
    batch = get_batch()
    while batch is None:
        generate_mode(rank=torch.distributed.get_rank())
        batch = get_batch()

    # 根据合成的数据根据GRPO算法进行模型训练
    loss = GRPO_step(batch)

    engine.backward(loss)
    engine.step()

    # 主进程打印日志
    if torch.distributed.get_rank() == 0:
        progress.set_description(f"Loss: {loss.item():.6f}")

    # save_steps保存模型pt
    if step % save_steps == 0:
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            print('saving ')
            save_name = f"./step_{step}"
            state_dict = engine.module.state_dict()
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            engine.module.save_pretrained(save_name, state_dict=state_dict)
            tokenizer.save_pretrained(save_name)
        dist.barrier()
