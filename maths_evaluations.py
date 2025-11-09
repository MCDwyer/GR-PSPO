import math, json, random, os
import sys
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from maths_rewards import gsm8k_numeric_reward, _extract_final_answer
from maths_dataset_loading import load_data, load_test_data

os.environ["WANDB_DISABLED"] = "true"

ROOT_DIR = "./cached_files"
BASE_MODEL_NAME="Qwen2.5-1.5B"

def _apply_chat(tokenizer, conversation):
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

def _bootstrap_ci(values: List[float], n_boot: int = 500, alpha: float = 0.05, rng: Optional[random.Random]=None):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = rng or random.Random(0)
    vals = np.array(values, dtype=np.float64)
    boot = []
    n = len(vals)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for __ in range(n)]
        boot.append(vals[idx].mean())
    lo = np.quantile(boot, alpha/2)
    hi = np.quantile(boot, 1 - alpha/2)
    return float(lo), float(hi)

def _pass_at_k_from_counts(c: int, n: int, k: int) -> float:
    if n == 0:
        return 0.0
    k = min(k, n)
    if (n - c) < k:
        return 1.0
    def logC(a, b):
        return (math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1))
    num = logC(n - c, k)
    den = logC(n, k)
    return 1.0 - math.exp(num - den)

def _prepare_inputs(tokenizer, prompt_text: str, device: torch.device):
    enc = tokenizer(prompt_text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}

def _decode_new_tokens(tokenizer, full_ids: torch.LongTensor, prompt_len: int) -> Tuple[str, int]:
    new_tokens = full_ids[0, prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, int(new_tokens.numel())

def _k_intervals(n_samples: int):
    ks, k = [1], 1
    while k < n_samples:
        k *= 2
        if k <= n_samples:
            ks.append(k)
    return ks

def _wilson_ci(p_hat: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0 or not (0.0 <= p_hat <= 1.0):
        return float("nan"), float("nan")
    # z for two-sided alpha (0.05 -> ~1.96)
    z = 1.959963984540054
    denom = 1.0 + (z*z)/n
    center = (p_hat + (z*z)/(2*n)) / denom
    half = z * math.sqrt((p_hat*(1.0 - p_hat) + (z*z)/(4*n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)

def _mean_ci(vals: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan")
    m = float(arr.mean())
    s = float(arr.std(ddof=1)) if n > 1 else 0.0
    # z for two-sided 95%
    z = 1.959963984540054
    half = z * (s / math.sqrt(max(1, n)))
    return m - half, m + half

def _collate_to_device(batch, pad_id, dev):
    # batch is a list of tuples: (sample, ids_1d, attn_1d)
    samples, ids_list, attn_list = zip(*batch)
    lens = [int(x.shape[0]) for x in ids_list]
    max_len = max(lens)
    B = len(ids_list)

    ids = torch.full((B, max_len), pad_id, dtype=ids_list[0].dtype)
    attn = torch.zeros((B, max_len), dtype=attn_list[0].dtype)
    # left-padding
    for b, (x, m) in enumerate(zip(ids_list, attn_list)):
        L = x.shape[0]
        ids[b, -L:] = x
        attn[b, -L:] = m

    return samples, ids.to(dev, non_blocking=True), attn.to(dev, non_blocking=True), lens

def _batched_multi_draws_generate(ids, attn, n_draws, do_sample, gen_kwargs, base_seed, dev):
    outs = []
    for d in range(n_draws):
        seq_seed = base_seed + d
        with torch.random.fork_rng(devices=[dev]):
            torch.manual_seed(seq_seed)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(seq_seed)
            out = model.generate(
                input_ids=ids,
                attention_mask=attn,
                do_sample=do_sample,
                num_return_sequences=1,
                **gen_kwargs,
            )  # shape: [B, seq_len_d]
        outs.append(out)
    return outs  # list length n_draws, each [B, seq_d]


def evaluate_temperature_sweep(
    model, tokenizer, test_dataset,
    temperatures=(0.0, 0.2, 0.4, 0.6, 0.8),
    samples_per_prompt=16,
    top_p=1.0,
    max_new_tokens=128,
    seed=123,
    save_path=None,
    batch_size=32,
):

    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Left-pad so batching works well for causal LM
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Cache tokenized prompts on CPU (1D tensors); decode later with original lengths
    cached = []
    for sample in test_dataset:
        prompt_text = _apply_chat(tokenizer, sample["conversation"])
        enc = tokenizer(prompt_text, return_tensors="pt", padding=False)
        # store 1D cpu tensors to keep memory modest; we move per-batch
        cached.append((sample, enc["input_ids"][0].cpu(), enc["attention_mask"][0].cpu()))

    k_vals = _k_intervals(samples_per_prompt)
    per_T, per_prompt_records = {}, {}

    for T in temperatures:
        do_sample = (T > 0.0)
        n_draws = samples_per_prompt if do_sample else 1

        per_prompt_correct_counts = []
        per_prompt_nsamples = []
        per_prompt_passk = defaultdict(list)
        per_prompt_lengths = []
        per_prompt_rewards_mean = []
        raw_rows = []

        for i in tqdm(range(0, len(cached), batch_size), desc=f"Eval T={T}"):
            chunk = cached[i:i + batch_size]
            samples, ids, attn, prompt_lens = _collate_to_device(
                batch=chunk,
                pad_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
                dev=device,
            )

            gen_kwargs = dict(
                max_new_tokens=int(max_new_tokens),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
                use_cache=True,
            )
            if do_sample:
                gen_kwargs.update(temperature=float(T), top_p=float(top_p))

            with torch.inference_mode():
                outs_per_draw = _batched_multi_draws_generate(
                    ids=ids,
                    attn=attn,
                    n_draws=(samples_per_prompt if do_sample else 1),
                    do_sample=do_sample,          # pass once here
                    gen_kwargs=gen_kwargs,        # no do_sample inside
                    base_seed=rng.randrange(10**12),
                    dev=device,
                )

            B = len(samples)
            batch_cutoff = ids.size(1)   # number of input tokens per row after left-padding

            for b in range(B):
                sample = samples[b]
                prompt_len = int(prompt_lens[b])
                completions, rewards, gen_lens = [], [], []

                for j in range(len(outs_per_draw)):
                    full = outs_per_draw[j][b:b+1]  # [1, seq_len_j]
                    text, new_len = _decode_new_tokens(tokenizer, full, batch_cutoff)
                    completions.append(text)
                    gen_lens.append(int(new_len))
                    rewards.append(gsm8k_numeric_reward(sample["gold_answer"], text))

                c = int(sum(1 if r > 0.5 else 0 for r in rewards))
                n = int(len(rewards))
                per_prompt_correct_counts.append(c)
                per_prompt_nsamples.append(n)
                per_prompt_lengths.append(float(np.mean(gen_lens)))
                per_prompt_rewards_mean.append(float(np.mean(rewards)))
                for k in k_vals:
                    per_prompt_passk[k].append(_pass_at_k_from_counts(c, n, k))

                raw_row = {
                    "temperature": float(T),
                    "prompt": sample["conversation"][1]["content"],
                    "gold_answer": sample["gold_answer"],
                    "completions": completions,
                    "rewards": rewards,
                    "gen_lens": gen_lens,
                    "extracted_gold_answer": _extract_final_answer(sample["gold_answer"]),
                }
                # keep optional fields if present
                if 'category' in sample:
                    raw_row['category'] = sample['category']
                if 'difficulty' in sample:
                    raw_row['difficulty'] = sample['difficulty']
                raw_rows.append(raw_row)

        # ----- aggregate (fast CIs) -----
        acc_vals = per_prompt_passk[1]
        acc_mean = float(np.mean(acc_vals)) if len(acc_vals) else float("nan")
        acc_ci = _wilson_ci(acc_mean, len(acc_vals)) if len(acc_vals) else (float("nan"), float("nan"))

        agg = {
            "accuracy_mean": acc_mean,
            "accuracy_ci95": acc_ci,
            "pass_at_k": {},
            "reward_mean": float(np.mean(per_prompt_rewards_mean)) if per_prompt_rewards_mean else float("nan"),
            "reward_ci95": _mean_ci(per_prompt_rewards_mean),
            "gen_len_tokens_mean": float(np.mean(per_prompt_lengths)) if per_prompt_lengths else float("nan"),
            "gen_len_tokens_ci95": _mean_ci(per_prompt_lengths),
        }
        for k in k_vals:
            vals = per_prompt_passk[k]
            m = float(np.mean(vals)) if vals else float("nan")
            agg["pass_at_k"][k] = {
                "mean": m,
                "ci95": _wilson_ci(m, len(vals)) if vals else (float("nan"), float("nan")),
            }

        per_T[float(T)] = agg
        per_prompt_records[float(T)] = {
            "correct_counts": per_prompt_correct_counts,
            "n_samples": per_prompt_nsamples,
            "pass_at_k_per_prompt": {k: per_prompt_passk[k] for k in k_vals},
            "mean_len_tokens_per_prompt": per_prompt_lengths,
            "mean_rewards_per_prompt": per_prompt_rewards_mean,
        }

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "a", encoding="utf-8") as f:
                for row in raw_rows:
                    f.write(json.dumps(row) + "\n")

    return {
        "temperatures": list(map(float, temperatures)),
        "samples_per_prompt": int(samples_per_prompt),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "metrics_by_T": per_T,
        "by_prompt_aux": per_prompt_records,
    }

def main(trained_model_path, samples_per_prompt):

    if "baseline" in trained_model_path:
        # Load base model & tokenizer
        tokenizer = AutoTokenizer.from_pretrained(f"{ROOT_DIR}/{BASE_MODEL_NAME}", use_fast=True)
        trained_model = AutoModelForCausalLM.from_pretrained(
            f"{ROOT_DIR}/{BASE_MODEL_NAME}",     
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()

    else:
        # Load fine tuned model & tokenizer
        tokenizer = AutoTokenizer.from_pretrained(trained_model_path, use_fast=True)
        trained_model = AutoModelForCausalLM.from_pretrained(
            trained_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    test_dataset_paths = ["gsm8k", "MATH-500", "SVAMP", "asdiv"]

    for test_dataset_name in test_dataset_paths:
        results_path = f"{trained_model_path}/{RESULTS_DIR}/{test_dataset_name}_raw_completions.jsonl"
        if os.path.exists(results_path):
            print(f"Skipping testing {trained_model_path} on {test_dataset_name} as data already exists")
            continue

        print(f"Testing {trained_model_path} on {test_dataset_name}")
        test_dataset = load_test_data(f"{ROOT_DIR}/{test_dataset_name}")

        print("type(test_dataset):", type(test_dataset))
        try:
            print("len(test_dataset):", len(test_dataset))
        except Exception as e:
            print("len(test_dataset) raised:", e)

        report = evaluate_temperature_sweep(
            model=trained_model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            temperatures=[0.0, 0.2, 0.4, 0.6, 0.8],
            samples_per_prompt=samples_per_prompt,
            top_p=1.0,
            max_new_tokens=128,
            seed=123,
            save_path=results_path,
        )

        for T, m in report["metrics_by_T"].items():
            print(f"T={T:.1f} len={m['gen_len_tokens_mean']:.1f}")
            k_value = 1
            while k_value <= samples_per_prompt:
                if k_value > samples_per_prompt:
                    break
                p1 = m["pass_at_k"][k_value]["mean"]
                print(f"\tp@{k_value} acc={p1:.3f}")
                k_value = int(2*k_value)

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_path> <samples_per_prompt>")
        sys.exit(1)

    model_path = sys.argv[1]
    samples_per_prompt = int(sys.argv[2])

    RESULTS_DIR = f'eval_{samples_per_prompt}_results'
    main(model_path, samples_per_prompt)
