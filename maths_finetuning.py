"""
Script to train and evaluate GRPO on GSM8K with different trust_region methods, like clipping, PSPO, or no clipping.
This script was used to train the models in the paper "It’s Not You, It’s Clipping: A Soft Trust-Region via Probability Smoothing for LLM RL" by Madeleine Dwyer, Adam Sobey, Adriane Chapman, 2025.
"""
from datasets import Dataset, load_from_disk, DatasetDict
from trl import SFTConfig, SFTTrainer
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback, TrainerCallback, DataCollatorForLanguageModeling
import pickle
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path
import wandb

from maths_rewards import gsm8k_numeric_reward
from maths_dataset_loading import load_data, load_test_data
from grpo_wrapper import GRPOTrainer, GRPOConfig

FORCE_RETRAIN = True
ROOT_DIR="./cached_files"

def _compute_loss_no_entropy(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):

    outputs = model(**{k: v for k, v in inputs.items() if k in ["input_ids","attention_mask","labels"]})
    # Standard causal LM returns .loss and .logits
    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
    return (loss, outputs) if return_outputs else loss

# Apply the patch (disables the entropy computation that causes the 4 vs 16 mismatch)
SFTTrainer.compute_loss = _compute_loss_no_entropy

def reward_function(completions, prompts, gold_answer, **kwargs):
    rewards = [gsm8k_numeric_reward(gold_ans, completion) for completion, gold_ans in zip(completions, gold_answer)]
    return rewards


def main(model, trust_region_method, seed):

    ft_model_path = f"{DIR_FOR_MODEL}/{trust_region_method}_{seed}"

    # wandb.login()
    run = wandb.init(
        project="gsm8k_training",
        name=f"{trust_region_method}_{seed}",
        reinit=True,
        mode="offline",
        settings=wandb.Settings(symlink=False)  # w/o this get weird saving issues when offline logging on iridis
    )

    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    model = f'{ROOT_DIR}/{model}'

    tokenizer = AutoTokenizer.from_pretrained(model)

    dataset_path = f"{ROOT_DIR}/gsm8k"

    if trust_region_method == "sft":
        os.environ["WANDB_DISABLED"] = "true"
        try:
            tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
            trained_model = AutoModelForCausalLM.from_pretrained(ft_model_path)
            print(f"Model loaded from: {ft_model_path}")
        except Exception:
            print(f"\nTraining with SFT for comparison: {trust_region_method}.")

            ds = load_data(dataset_path, tokenizer, seed=seed, sft=True)
            train_ds, val_ds = ds["train"], ds["validation"]
            print(f"Loaded dataset with {len(train_ds)} examples")

            def sft_preprocess(ex):

                text = tokenizer.apply_chat_template(ex["conversation"], tokenize=False, add_generation_prompt=False)
                
                toks = tokenizer(
                    text,
                    truncation=True,
                    max_length=1024,
                    padding=False,                 # let collator pad
                    return_attention_mask=True,
                )
                return toks

            # Build a response template that exactly matches your chat template's assistant prefix.
            assistant_prefix = tokenizer.apply_chat_template(
                [{"role":"assistant","content":""}],
                tokenize=False,
                add_generation_prompt=False,
            )

            train_ds = train_ds.map(sft_preprocess, remove_columns=train_ds.column_names, batched=False)
            val_ds = val_ds.map(sft_preprocess, remove_columns=val_ds.column_names, batched=False)

            # Use token IDs to be robust to special tokens
            response_template_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)

            sft_config = SFTConfig(
                # === I/O ===
                output_dir=ft_model_path,
                save_steps=100,
                seed=seed,

                # === precision (V100) ===
                fp16=False,
                bf16=True,
                gradient_checkpointing=False,
                num_train_epochs=20,

                # === batching ===
                per_device_train_batch_size=4,
                gradient_accumulation_steps=16,

                packing=False,                 # <- turn packing off to avoid 4 vs 16 mismatch
                prediction_loss_only=True,

                # === eval ===
                do_eval=True,
                eval_strategy="steps",
                eval_steps=100,
                metric_for_best_model="eval_loss",
                load_best_model_at_end=True,
                report_to=['wandb'],
                completion_only_loss=True, 
            )

            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_ds,
                eval_dataset=val_ds,
            )

            dl = trainer.get_train_dataloader()
            batch = next(iter(dl))
            for k, v in batch.items():
                if hasattr(v, "shape"):
                    print(k, tuple(v.shape))

            trainer.train()

            trained_model = trainer.model
            tokenizer = trainer.tokenizer

            try:
                # Save to local directory
                trained_model.save_pretrained(ft_model_path)
                tokenizer.save_pretrained(ft_model_path)
            except Exception:
                print("Saving models failed")

            return

    ds = load_data(dataset_path, tokenizer, seed=seed)
    train_ds, val_ds = ds["train"], ds["validation"]

    print(len(train_ds))

    if "0.5B" in model:
        if trust_region_method == "clip":
            lr = 1e-6
            epsilon = 0.1
        if trust_region_method == "noclip":
            lr = 1e-6
            epsilon = 0.0
            trust_region_method = "clip"  # use clip trainer with 0.0 epsilon to avoid errors, number of iterations will be set to 1 so no clipping will occur
        else:
            lr = 5e-7
            epsilon = 0.0
            smoothing_alpha = 0.1
    elif "1.5B" in model:
        if trust_region_method == "clip":
            lr = 5e-07
            epsilon = 0.2
        if trust_region_method == "noclip":
            lr = 1e-06
            epsilon = 0.0
            trust_region_method = "clip"  # use clip trainer with 0.0 epsilon to avoid errors
        else:
            lr = 5e-07
            epsilon = 0.0
            smoothing_alpha = 0.1

    loss_type = "dapo"
    
    training_args = GRPOConfig(
        # # === I/O ===
        output_dir=ft_model_path,
        save_steps=100,
        seed=seed,
        num_iterations=2 if trust_region_method != "noclip" else 1,
        gradient_checkpointing=False,
        report_to=["wandb"],
        run_name=f"{trust_region_method}_{seed}",

        # === precision (V100) ===
        fp16=False, 
        bf16=True,
        num_generations = 4,
        per_device_train_batch_size=4,  
        gradient_accumulation_steps=16,  # Increase to maintain effective batch size, default was 8 and 8, so this (4 and 16) is still effective batch size of 64?
        remove_unused_columns=False,
        max_completion_length=128,         # equiv. to max_new_tokens
        temperature=0.8,                   # keep answers concise - higher temperature leads to more exploration, default was 1.0 -> rescales the logits (raw scores) before applying softmax
        top_p=0.9,                         # what tokens to sample from, 1.0 sample from all, 0.8 samples from top 0.8 so removes the tail - nucleus sampling -> instead of sampling from all tokens, keep only the smallest set whose cumulative probability ≥ p    
        beta=0.0,
        warmup_steps=max(1, int(0.02*len(train_ds))),
        loss_type=loss_type,

        learning_rate=lr,
        epsilon=epsilon,

        # eval
        do_eval=True,
        eval_strategy="steps",
        eval_steps=100,
        metric_for_best_model="eval_reward",
        load_best_model_at_end=True,

        trust_region_method=trust_region_method,
        smoothing_alpha=smoothing_alpha,
    )

    try:
        if not FORCE_RETRAIN:
            tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
            trained_model = AutoModelForCausalLM.from_pretrained(ft_model_path)
            print(f"Model loaded from: {ft_model_path}")
        else:
            print(f"forcing retraining")
            raise Exception
        
    except Exception:
        print(f"\nTraining with default grpo (no clipping) for comparison: {trust_region_method}.")
        print(f"Loaded dataset with {len(train_ds)} examples")
        
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_function,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )

    trainer.train()

    trained_model = trainer.model
    tokenizer = trainer.tokenizer

    try:
        trained_model.save_pretrained(ft_model_path)
        tokenizer.save_pretrained(ft_model_path)
    except Exception:
        print("Saving models failed")


if __name__=="__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <base_model> <trust_region_method> <seed>")
        print("trust_region_method: clip, pspo, noclip")
        sys.exit(1)
        
    model = sys.argv[1]    
    trust_region_method = sys.argv[2]
    seed = int(sys.argv[3])

    DIR_FOR_MODEL=f'{ROOT_DIR}/gsm8k_{model}_ft_models'

    main(model, trust_region_method, seed)
