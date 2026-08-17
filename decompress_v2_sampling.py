import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
# import setproctitle
from utils import init_by_config_path

import itertools
import math
from typing import List, Sequence, Dict, Any

import numpy as np
from multiprocessing import Pool

import random

import random, math

def hamming_bits(a, b):
    return sum(x != y for x, y in zip(a, b))


def mh_subset_select(candidates, K=10, steps=3000, lam=0.1, beta=1.0, seed=0):

    random.seed(seed)
    N = len(candidates)

    scores = [c["path_prob"] for c in candidates]
    bitstreams = [c["bits"] for c in candidates]

    def energy(S):
        score_sum = sum(scores[i] for i in S)
        div = sum(
            hamming_bits(bitstreams[i], bitstreams[j])
            for idx, i in enumerate(S) for j in S[idx + 1:]
        )
        return -(score_sum + lam * div)

    S = sorted(range(N), key=lambda i: scores[i], reverse=True)[:K]
    E = energy(S)

    for _ in range(steps):
        out = random.choice(S)
        inn = random.choice([i for i in range(N) if i not in S])
        S2 = [i for i in S if i != out] + [inn]
        E2 = energy(S2)

        if random.random() < min(1.0, math.exp(-beta * (E2 - E))):
            S, E = S2, E2

    return [candidates[i] for i in S]

def build_flipped_bits(orig_bits: Sequence[int],
                       flip_indices: Sequence[int]) -> List[int]:
    """
    根据 flip_indices 在 orig_bits 上翻转对应位置，返回新的比特序列。
    flip_indices 是下标（0-based）。
    """
    bits = list(orig_bits) 
    for idx in flip_indices: 
        bits[idx] = 1 - bits[idx] 
    return bits

def precompute_contrib(bit_probs: Sequence[float]):
    p = np.asarray(bit_probs, dtype=np.float64)
    delta = np.abs(p - 0.5)
    keep = 2.0 * delta               # 不翻该位时的贡献
    flip = 1.0 - 2.0 * delta         # 翻该位时的贡献

    diff = flip - keep               # 翻 vs 不翻 的“增量”
    base_sum = keep.sum()         # 所有位都“不翻”时的 log 概率和
    return diff, base_sum


def compute_path_bit_score_fast(diff: np.ndarray,
                                base_sum: float,
                                L: int,
                                flip_indices: Sequence[int]) -> float:
    if not flip_indices:
        score_sum = base_sum
        # score_log = base_sum
    else:
        # 这里只对翻转的位做求和，复杂度 O(#flip_indices)
        score_sum = base_sum + float(np.sum(diff[list(flip_indices)]))
        # score_log = base_sum + float(np.sum(diff[list(flip_indices)]))
    return score_sum / L
    # return score_log

def decode_one_path(bits, trainer, out_file=None):
    precision = config.ac.precision
    top_value = 2 ** precision
    quarter = top_value // 4
    half = 2 * quarter
    low = 0
    high = top_value

    # ---- 1) 初始化 z ----
    z = 0
    i = 1  # bit 下标，从 1 开始只是沿用你原来的写法
    while i <= precision and i <= len(bits):
        if bits[i - 1] == 1:
            z = z + 2 ** (precision - i)
        i += 1

    # ---- 2) 初始化解码状态 ----
    output = [tokenizer.bos_token_id]
    last_token_idx = None
    repeat_count = 0
    log_likelihood = 0.0
    decoded_token = []

    # ---- 3) 主循环：逐 token 做算术解码 ----
    while (output[-1] != tokenizer.eos_token_id and z != half) or i <= precision + 1:
        # 3.1 调 LLM 得到下一个 token 的概率分布
        logits = trainer.predict_step(output[-max_len:])
        logits = logits.detach()
        probs = F.softmax(logits[0][-1], dim=-1)
        probs = probs.to(torch.float64)
        # 3.2 构造累积分布 & 区间
        cumprobs = torch.cumsum(probs, dim=0)
        cumprobs = torch.cat((torch.tensor([0.0], device=probs.device), cumprobs), dim=0)
        cumprobs = cumprobs.to(torch.float64)

        width = high - low + 1
        widths = width * cumprobs[1:]  # 对应每个 symbol 的右端点比例
        lows = low + widths[:-1].to(torch.int64)
        highs = low + widths[1:].to(torch.int64)

        # 3.3 找到 z 所落在的唯一区间（对应某个 symbol_idx）
        valid_indices = (lows <= z) & (z < highs)
        if not valid_indices.any():
            # 算术解码失败，说明这条 bits 序列在当前模型下不合法
            return "", -1e30

        # 0-based 符号索引（和 probs 的下标对齐）
        symbol_idx = valid_indices.nonzero(as_tuple=True)[0][0].item()

        tgt_idx = symbol_idx + 1

        # 3.4 累加 LLM log 概率
        # log_likelihood += torch.log(probs[symbol_idx]).item()
        log_likelihood += torch.log(probs[tgt_idx]).item()

        # 3.5 记录 token
        token_str = tokenizer.decode([tgt_idx])  # 解成字符串
        decoded_token.append(token_str)

        if out_file is not None and tgt_idx != tokenizer.eos_token_id:
            safe_token_str = token_str.replace('\n', '<unk>')
            out_file.write(safe_token_str)

        # if tgt_idx != tokenizer.eos_token_id: 
            # decoded_token = decoded_token.replace('\n', '<unk>')
            # out_file.write(decoded_token)
        # print(f'Decoded token: {decoded_token} (id={tgt_idx})')

        if tgt_idx == last_token_idx:
            repeat_count += 1
        else:
            repeat_count = 1
        if repeat_count >= 3:

            break
        last_token_idx = tgt_idx

        # 3.6 更新 output / low / high / z / i（这里和原 decompress 保持一致）
        _output = output + [tgt_idx]
        _out_len = min(len(output), max_len)
        _y = _output[-_out_len:]  
        logits = logits.detach()

        output = _output
        low = lows[tgt_idx - 1]
        high = highs[tgt_idx - 1]

        # renormalization loop 1
        while high < half or low >= half:
            if high < half:
                low = 2 * low
                high = 2 * high
                z = 2 * z
            elif low >= half:
                low = 2 * (low - half)
                high = 2 * (high - half)
                z = 2 * (z - half)
            if i <= len(bits) and bits[i - 1] == 1:
                z += 1
            i += 1

        # renormalization loop 2
        while low >= quarter and high < 3 * quarter:
            low = 2 * (low - quarter)
            high = 2 * (high - quarter)
            z = 2 * (z - quarter)
            if i <= len(bits) and bits[i - 1] == 1:
                z += 1
            i += 1

    # ---- 4) 把 token id 转回文本 ----

    if out_file is not None:
        out_file.write('\n')
        out_file.flush()
    
    return decoded_token, log_likelihood



import itertools
import heapq
from typing import List, Sequence, Dict, Any

import numpy as np
import heapq
import itertools
from typing import List, Dict, Any, Sequence


def generate_all_flip_paths_topk1(
    orig_bits: Sequence[int],
    bit_probs: Sequence[float],
    top_k: int = 10,
    max_drop: int = 5,      
    max_subset: int = 5,
    toggle_cap: int = 10,   
    fixed_prefix_len: int = 0,
) -> List[Dict[str, Any]]:

    L = len(orig_bits)
    assert L == len(bit_probs), "orig_bits 和 bit_probs 长度不一致"

    # 预计算 diff & base_sum：
    # diff[i] = flip_i - keep_i，base_sum = 所有位都“不翻”时的总贡献
    diff, base_sum = precompute_contrib(bit_probs)
    diff = np.asarray(diff, dtype=np.float64)

    # 候选集合：key = tuple(sorted(flip_indices))
    candidates: Dict[tuple, Dict[str, Any]] = {}

    def add_candidate(flip_indices) -> None:
        """
        把一条翻转方案加入候选集合（自动去重，只保留更高分的版本）。
        flip_indices 为 0-based 下标序列 / 集合。
        """
        flip_filtered = [i for i in set(flip_indices) if i >= fixed_prefix_len]
        # flip_sorted = sorted(set(flip_indices))
        flip_sorted = sorted(flip_filtered)
        key = tuple(flip_sorted)

        score = compute_path_bit_score_fast(diff, base_sum, L, flip_sorted)

        old = candidates.get(key)
        if (old is None) or (score > old["path_prob"]):
            bits_path = build_flipped_bits(orig_bits, flip_sorted)
            candidates[key] = {
                "bits":         bits_path,
                "flip_indices": flip_sorted,
                "path_prob":    score,
            }

    # ---- 0) baseline：完全不翻 ----
    add_candidate([])

    # ---- 1) 基准最优路径 B：翻转所有 diff>0 的位 ----
    base_flip_idx_all = np.where(diff > 0)[0]
    base_flip_idx = base_flip_idx_all[base_flip_idx_all >= fixed_prefix_len]
    base_flip_set = set(int(i) for i in base_flip_idx)
    if base_flip_set:
        add_candidate(base_flip_set)

    # ---- 2) 按 |diff| 升序选出若干“可扰动位” ----
    abs_diff = np.abs(diff)
    idx_sorted_all = np.argsort(abs_diff)          # 从 |diff| 最小到最大
    idx_sorted_suffix = [int(i) for i in idx_sorted_all if i >= fixed_prefix_len]
    M_cap = min(len(idx_sorted_suffix), toggle_cap)
    toggle_candidates = idx_sorted_suffix[:M_cap]

    # ---- 3) 在这些可扰动位上枚举 T（|T| <= max_subset），构造 B ⊕ T ----
    max_r = min(max_subset, M_cap)
    for r in range(0, max_r + 1):
        for combo in itertools.combinations(toggle_candidates, r):
            combo_set = set(combo)
            # flips = B ⊕ T（对称差）
            flips_set = base_flip_set ^ combo_set
            add_candidate(flips_set)

    # ---- 4) 按 path_prob 从大到小排序，取前 top_k ----
    cand_list = list(candidates.values())
    cand_list.sort(key=lambda d: d["path_prob"], reverse=True)

    top20 = cand_list[:top_k]                     

    top10 = mh_subset_select(
        top20,
        K=10,                                
        steps=500,                           
        lam=0.09,                              
        beta=1.0,                           
        seed=42                          
    )
    return top10


def search_best_path_with_llm(orig_bits: Sequence[int],
                              bit_probs: Sequence[float],
                              trainer,
                              top_k: int = 10,fixed_prefix_len: int = 0) -> Dict[str, Any]:

    # 1) 穷举 0~L 位翻转，按比特侧打分取前 top_k 条路径
    candidates = generate_all_flip_paths_topk1(orig_bits, bit_probs, top_k=top_k, fixed_prefix_len=fixed_prefix_len,)
    print(f'length of candidates:{len(candidates)}')
    # print(f'Generated candidate paths: {candidates}')
    results = []
    for cand in candidates:
        bits_cand   = cand["bits"]
        path_prob   = cand["path_prob"]
        flips       = cand["flip_indices"]

        # 2) 用 LLM 对该路径做算术解码
        decoded_text, llm_logp = decode_one_path(bits_cand, trainer, out_file=None)

        # 3) “路径概率 × LLM 似然” -> 在 log 域是相加
        final_log_score = 0.001 * llm_logp + 0.999 * math.log(path_prob + 1e-12)


        results.append({
            "bits": bits_cand,
            "flip_indices": flips,
            "path_prob": path_prob,
            "llm_logp": llm_logp,
            "final_log_score": final_log_score,
            "text": decoded_text,
        })
    # 4) 按最终 log 分数选出最佳路径
    best = max(results, key=lambda d: d["final_log_score"])

    best_result = {
        "best_bits":         best["bits"],
        "best_flip_indices": best["flip_indices"],
        "best_path_prob":    best["path_prob"],
        "best_llm_logp":     best["llm_logp"],
        "best_final_score":  best["final_log_score"],
        "best_text":         best["text"],
        "top_candidates":    results,
    }
    return best_result

from dataclasses import dataclass
import math

def load_bit_probs(path):
    """
    一行一行读取 prob 文件，返回一个 list，
    其中每个元素是该行的概率列表：
        all_probs[line_idx] -> [p_0, p_1, ..., p_{L-1}]
    注意：不同 line 的长度可以不一样。
    """
    all_probs = []

    with open(path, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue  # 空行跳过

            # 当前这一行的所有概率
            str_vals = line.split()
            probs = [float(v) for v in str_vals]

            all_probs.append(probs)   # 一行一个 list

    if not all_probs:
        raise ValueError(f"文件为空或没有有效行：{path}")

    return all_probs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(42)
    root_prob = 'ECCT_prob'
    root_dir_in = 'ECCT_extracted'
    root_dir_out = 'ECCT_restored'
    N, K = 49, 24

    code_name = f'LDPC_K{K}_N{N}'
    channel = 'AWGN'  # AWGN or Rayleigh
    filename = 'demo_decode'



    number = None

    for SNR in tqdm(range(-3,16,3)):
        parser = argparse.ArgumentParser(description='Decompress a binary file while training an LLM.')
        parser.add_argument('--input_file', type=str, \
            default=f'mydata/{root_dir_in}/{code_name}/{channel}/{filename}_SNR_{SNR}.txt')
        parser.add_argument('--output_file', type=str, \
            default=f'mydata/{root_dir_out}/{code_name}/{channel}/{filename}_SNR_{SNR}.txt')

        parser.add_argument('--config_file', type=str, default='config/global/demo.yaml')
        args = parser.parse_args()
        
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        out_file = open(args.output_file, 'w', encoding='utf-8')
        
        prob_path = f"mydata/{root_prob}/demo/{code_name}/{channel}/demo_prob_SNR_{SNR}.txt"
        bit_probs_all = load_bit_probs(prob_path)

        config, tokenizer, trainer, io = init_by_config_path(
            args.input_file, args.output_file, args.config_file, 'decompress'
        )
        max_len = config.block_size
        print('Start decompressing with the following config...')
        print(config)
        
        with open(args.input_file, 'r') as f:
            bit_streams = f.read().splitlines()  # Split into bitstream lists by line
        
        
        token_num, bits_num = 0, 0

        for line_idx, (bit_string, bit_probs) in enumerate(zip(bit_streams, bit_probs_all), start=1):
            bits = [int(bit) for bit in bit_string]

            if len(bits) != len(bit_probs):
                print(f"[WARN] line {line_idx}: len(bits)={len(bits)}, len(bit_probs)={len(bit_probs)} 不一致")

            best = search_best_path_with_llm(bits, bit_probs, trainer, top_k=20, fixed_prefix_len= number[line_idx-1],)
            # print(best)
            
            best_bits = best["best_bits"]
            
            decoded_tokens, _ = decode_one_path(best_bits, trainer, out_file=out_file)
            

