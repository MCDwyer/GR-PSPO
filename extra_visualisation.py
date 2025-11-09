"""
Mixin class to add visualizations/logging to TRL trainers, and run simple training in same way as was for the main running of the experiment.
"""
from dataclasses import dataclass
from collections import deque
import contextlib
import torch
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px       
import numpy as np                
import pickle
import wandb

STEP_SIZE = 25
trust_region_method = "pspo"

@dataclass
class VisCfg:
    log_every: int = STEP_SIZE             
    probe_every: int = STEP_SIZE 
    probe_batch_size: int = 8
    probe_max_tokens: int = 256
    slice_span: float = 0.5
    slice_points: int = 41
    beta_sweep: tuple = (0.0, 0.01, 0.02, 0.05, 0.1)
    sharp_iters: int = 12
    lora_only: bool = False
    seed: int = 0

class VisualizeTRL:
    """
    Mixin: stash intermediates on the trainer and log them via log().
    You only need to call self._vis_stash(inter, inputs) once inside compute_loss.
    """
    # in your VisualizeTRL mixin

    def _vprint(self, *args, **kwargs):
        """Rank-0 safe print."""
        acc = getattr(self, "accelerator", None)
        if acc is not None and hasattr(acc, "print"):
            try:
                acc.print(*args, **kwargs)
                return
            except Exception:
                pass
        print(*args, **kwargs)

    def _vis_setup(self):
        # Call this once in your Trainer.__init__()
        self._vis_cfg = getattr(self.args, "vis_cfg", VisCfg())
        self._vis_last = None              # latest intermediates (CPU tensors)
        self._vis_probe_batch_cpu = None   # lazily captured small batch
        self._vis_warned_wandb = False

    def _trim_for_probe(self, v):
        # Keep form but shrink size. Works for 0D, 1D and >=2D tensors.
        if not torch.is_tensor(v):
            return v
        if v.ndim == 0:
            return v.detach().cpu()  # scalar; do NOT slice
        if v.ndim == 1:
            return v[: self._vis_cfg.probe_batch_size].detach().cpu()
        # assume (B, T, ...) — trim B then T
        return v[: self._vis_cfg.probe_batch_size, : self._vis_cfg.probe_max_tokens, ...].detach().cpu()

    # === called from compute_loss ===
    def _vis_stash(self, inter: dict, inputs: dict):
        
        """Detach + store just what we need for plots. No return value."""
        def d(x): return x.detach().cpu() if torch.is_tensor(x) else x
        # keep lightweight tensors only
        keep = {
            "coef_1": d(inter["coef_1"]),                         # PPO ratio
            "coef_2": d(inter["coef_2"]),
            "entropies": d(inter["entropies"]),
            "completion_mask": d(inter["completion_mask"]),
            "advantages": d(inter["advantages"]),
            "is_low_clipped": d(inter["is_low_clipped"]),
            "is_high_clipped": d(inter["is_high_clipped"]),
        }
        if "per_token_kl" in inter:
            keep["per_token_kl"] = d(inter["per_token_kl"])
        self._vis_last = keep

        if self._vis_probe_batch_cpu is None:
            probe = {}
            for k, v in inputs.items():
                try:
                    probe[k] = self._trim_for_probe(v)
                except Exception as e:
                    # optional: skip troublesome keys rather than fail
                    # self._vprint(f"[vis] skipping key {k}: {e}")
                    continue

            # Ensure num_items_in_batch exists and is scalar (int or 0D tensor)
            if "num_items_in_batch" not in probe:
                # Best-effort: use local batch size from any batched field
                bsz = None
                for kk in ("completion_ids", "prompt_ids", "attention_mask", "prompt_mask", "completion_mask"):
                    t = probe.get(kk, None)
                    if torch.is_tensor(t) and t.ndim >= 1:
                        bsz = int(t.shape[0]); break
                if bsz is None:
                    bsz = 1
                probe["num_items_in_batch"] = torch.tensor(bsz, dtype=torch.int64)

            # If num_items_in_batch came in as a Python int, make it a 0D tensor later
            self._vis_probe_batch_cpu = probe


    def log(self, logs, *args, **kwargs):
        """
        Compatible with both HF Trainer (log(logs)) and TRL (log(logs, step)).
        We forward all extra args/kwargs to super().log and compute the step
        from kwargs or from self.state.global_step if not provided.
        """
        # Extract step if TRL passed it
        step = None
        if len(args) >= 1:
            step = args[0]
        if "step" in kwargs and kwargs["step"] is not None:
            step = kwargs["step"]

        # Call parent logging with original signature
        try:
            super().log(logs, *args, **kwargs)
        except TypeError:
            # Fallback for HF Trainer (no step argument)
            super().log(logs)

        # If W&B isn't available, stop here
        try:
            import wandb  # local import to avoid import-time errors
        except Exception:
            if not getattr(self, "_vis_warned_wandb", False):
                self._vis_warned_wandb = True
                self._vprint("W&B not installed; skipping visual logs.")
            return

        # Determine the step to use for visual logs
        if step is None:
            step = int(getattr(self.state, "global_step", 0))

        # Cheap, per-step logs
        if self._vis_last is not None and step % self._vis_cfg.log_every == 0:
            self._vis_log_step(self._vis_last, step)

        # Heavier probes on cadence
        if self._vis_probe_batch_cpu is not None and step % self._vis_cfg.probe_every == 0:
            self._vis_probe_heatmaps(step)
            self._vis_probe_slices(step)
            self._vis_probe_beta_frontier(step)

            try:
                self._vis_probe_sharpness(step)  # inside your log()
            except RuntimeError as e:
                if "efficient_attention_backward is not implemented" in str(e):
                    # Log a sentinel and move on
                    try: import wandb; wandb.log({"probe/top_hessian_eig": float("nan")}, step=step)
                    except Exception: pass
                else:
                    raise

    # ----------------- cheap per-step logging -----------------
    def _vis_log_step(self, stash: dict, step: int, tag="train"):
        cm = stash["completion_mask"].bool()
        ratio = stash["coef_1"]
        ent   = stash["entropies"]
        adv   = stash["advantages"].unsqueeze(1).expand_as(ratio).to(ratio.dtype)
        low   = stash["is_low_clipped"].float()
        high  = stash["is_high_clipped"].float()

        log = {
            f"{tag}/ratio/mean": ratio[cm].mean().item(),
            f"{tag}/adv/mean":   adv[cm].mean().item(),
            f"{tag}/entropy/mean": ent[cm].mean().item(),
            f"{tag}/clip/low_mean": low[cm].mean().item(),
            f"{tag}/clip/high_mean": high[cm].mean().item(),
            f"{tag}/ratio/hist": wandb.Histogram(ratio[cm].numpy()),
            f"{tag}/adv/hist":   wandb.Histogram(adv[cm].numpy()),
            f"{tag}/entropy/hist": wandb.Histogram(ent[cm].numpy()),
        }
        if "per_token_kl" in stash:
            log[f"{tag}/kl/mean"] = stash["per_token_kl"][cm].mean().item()
        wandb.log(log, step=step)

    # ----------------- probe helpers -----------------
    def _to_device_probe(self):
        out = {}
        for k, v in self._vis_probe_batch_cpu.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.accelerator.device)
            elif k == "num_items_in_batch":
                out[k] = torch.tensor(v, device=self.accelerator.device)  # if it was an int
            else:
                out[k] = v

        out = self._vis_ensure_ref_logps(out)
        return out

    def _base_compute_loss(self, model, inputs):
        # Resolve to the next class in MRO after the mixin (e.g., GRPOTrainer)
        return super(VisualizeTRL, self)._compute_loss(model, inputs)


    @torch.no_grad()
    def _vis_probe_heatmaps(self, step: int, tag="probe"):
        b = self._to_device_probe()
        self.model.eval()
        _ = self._base_compute_loss(self.model, b)    # new
        inter = self._vis_last
        cm = inter["completion_mask"]
        figs = []
        figs.append(self._heatmap(inter["coef_1"], cm, "PPO ratio"))
        figs.append(self._heatmap(inter["coef_2"], cm, "Clipped/Smoothed PPO ratio"))
        figs.append(self._heatmap(inter["entropies"], cm, "Entropy"))
        if "per_token_kl" in inter:
            figs.append(self._heatmap(inter["per_token_kl"], cm, "Per-token KL"))
        clip_map = inter["is_low_clipped"].float() + 2*inter["is_high_clipped"].float()
        figs.append(self._heatmap(clip_map, cm, "Clip map (1=low, 2=high)"))
        
        # Change: Log the Plotly figure objects directly
        # log_dict = {f"{tag}/heatmap/{i}": fig for i, fig in enumerate(figs)}
        # wandb.log(log_dict, step=step)

        log_dict = {f"{tag}/heatmap/{i}": wandb.Plotly(fig) for i, fig in enumerate(figs)}
        wandb.log(log_dict, step=step)
        

    def _heatmap(self, tensor, mask, title):
        # 1. Apply mask: Plotly prefers NaN for masked values
        t_masked = tensor.masked_fill(~mask.bool(), float("nan"))
        
        # 2. Convert to numpy array
        data = t_masked.numpy() 

        # 3. Create Plotly figure
        fig = go.Figure(data=[
            go.Heatmap(
                z=data,
                colorscale='Viridis', # Common colorscale
                connectgaps=False      # Don't interpolate NaN/masked values
            )
        ])

        # 4. Update layout for title and axis labels
        fig.update_layout(
            title=title,
            xaxis_title="Token Index",
            yaxis_title="Sequence Index",
            autosize=True,
            margin=dict(l=20, r=20, t=40, b=20),
            height=300, # Set a reasonable default height
            width=600
        )
        return fig

    # ---- parameter utils for slices/curvature (no return changes) ----
    def _flat_params(self):
        return torch.cat([p.detach().reshape(-1) for p in self.model.parameters()])

    def _assign_params(self, flat):
        i=0
        for p in self.model.parameters():
            n=p.numel()
            p.data.copy_(flat[i:i+n].view_as(p)); i+=n

    def _make_dir(self, seed: int):
        """Version-safe random direction with per-layer normalization.
        Respects lora_only if set."""
        device = self.accelerator.device
        blocks = []

        # Try modern path: randn with generator (works even if randn_like doesn't)
        try:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
            for n, p in self.model.named_parameters():
                if self._vis_cfg.lora_only and ".lora_" not in n:
                    blocks.append(torch.zeros_like(p))
                    continue
                # Use torch.randn (NOT randn_like) with generator
                r = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                r = r / r.norm().clamp_min(1e-12)
                blocks.append(r)
        except TypeError:
            # Older PyTorch: no generator kw support; scope RNG with fork_rng
            with torch.random.fork_rng(devices=[device], enabled=True):
                torch.manual_seed(int(seed))
                try:
                    torch.cuda.manual_seed_all(int(seed))  # no-op on CPU-only
                except Exception:
                    pass
                for n, p in self.model.named_parameters():
                    if self._vis_cfg.lora_only and ".lora_" not in n:
                        blocks.append(torch.zeros_like(p))
                        continue
                    r = torch.randn_like(p)  # no generator kw
                    r = r / r.norm().clamp_min(1e-12)
                    blocks.append(r)

        d = torch.cat([b.reshape(-1) for b in blocks])
        return d / d.norm().clamp_min(1e-12)

    @contextlib.contextmanager
    def _param_sandbox(self):
        theta0 = self._flat_params().clone()
        try:
            yield theta0
        finally:
            self._assign_params(theta0)

    @torch.no_grad()
    def _vis_ensure_ref_logps(self, batch: dict):
        """
        Ensure batch['ref_per_token_logps'] exists (B, T_completion).
        Uses the trainer's ref model. Works for text-only and passes through any
        image keys already in the batch.
        """
        if "ref_per_token_logps" in batch and torch.is_tensor(batch["ref_per_token_logps"]):
            return batch # already present

        # Find the reference model attribute used by your TRL version.
        ref = getattr(self, "ref_model", None) \
            or getattr(self, "reference_model", None) \
            or getattr(self, "policy_reference_model", None)
        if ref is None:
            # Last-resort: create a zero tensor with the right shape so parent loss doesn't crash.
            # (Better than crashing; but real values are preferred if a ref exists.)
            comp = batch["completion_ids"]
            batch["ref_per_token_logps"] = torch.zeros(
                comp.shape[0], comp.shape[1], device=comp.device, dtype=torch.float32
            )
            return batch

        # Build concatenated inputs (prompt + completion)
        prompt_ids, prompt_mask = batch["prompt_ids"], batch["prompt_mask"]
        completion_ids, completion_mask = batch["completion_ids"], batch["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Pass through any multimodal fields if your trainer uses them
        kwargs = {}
        for k in ("pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes"):
            if k in batch: kwargs[k] = batch[k]

        # Reuse the same utility your loss uses
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            ref,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=False,
            **kwargs,
        )
        batch["ref_per_token_logps"] = per_token_logps

        return batch

    def _vis_probe_sharpness(self, step:int, tag="probe"):
        device = self.accelerator.device
        b = self._to_device_probe()

        # params list (only those that require grad)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return

        # helper: Loss with graph enabled
        def loss_with_graph():
            self.model.train(False)
            return self._base_compute_loss(self.model, b)  

        # power iteration
        theta_dim = sum(p.numel() for p in params)
        v = torch.randn(theta_dim, device=device)
        v = v / v.norm().clamp_min(1e-12)

        def hvp(v_vec):
            # 1) grads wrt params
            L = loss_with_graph()
            grads = torch.autograd.grad(L, params, create_graph=True, allow_unused=True)
            # coalesce Nones to zeros
            flat_g = []
            for g, p in zip(grads, params):
                if g is None:
                    flat_g.append(torch.zeros_like(p).reshape(-1))
                else:
                    flat_g.append(g.reshape(-1))
            flat_g = torch.cat(flat_g)

            # 2) dot(flat_g, v) and grad again wrt params
            dot = (flat_g * v_vec).sum()
            Hv_parts = torch.autograd.grad(dot, params, retain_graph=False, allow_unused=True)

            flat_Hv = []
            for h, p in zip(Hv_parts, params):
                if h is None:
                    flat_Hv.append(torch.zeros_like(p).reshape(-1))
                else:
                    flat_Hv.append(h.reshape(-1))
            return torch.cat(flat_Hv).detach()

        iters = int(getattr(self._vis_cfg, "sharp_iters", 12))
        for _ in range(iters):
            Hv = hvp(v)
            norm = Hv.norm().clamp_min(1e-12)
            v = (Hv / norm)

        # final Rayleigh quotient λ ≈ vᵀ H v
        Hv = hvp(v)
        lam = float((v * Hv).sum().item())

        try:
            import wandb
            wandb.log({f"{tag}/top_hessian_eig": lam}, step=step)
            
        except Exception:
            pass



    # ----------------- slices -----------------
    @torch.no_grad()
    def _vis_probe_slices(self, step: int, tag="probe"):
        b = self._to_device_probe()

        def scalar_loss():
            out = self._base_compute_loss(self.model, b)   # new
            return float(out.detach().cpu())

        with self._param_sandbox() as theta0:
            d1 = self._make_dir(self._vis_cfg.seed)
            d2 = self._make_dir(self._vis_cfg.seed+1)
            span = self._vis_cfg.slice_span * theta0.norm().item()
            A = torch.linspace(-span, span, self._vis_cfg.slice_points, device=self.accelerator.device)

            # --- 1-D Slice (Line Plot) ---
            vals=[]
            for a in A:
                self._assign_params(theta0 + a*d1)
                vals.append(scalar_loss())
            
            # Use Plotly Express for a simple line plot
            fig_1d = px.line(x=A.tolist(), y=vals, labels={'x':'alpha', 'y':'Loss'}, 
                             title="1-D Frozen Loss Slice")
            
            # Change: Log the Plotly figure directly (replaces wandb.plot.line_series)
            # wandb.log({"probe/slice_1d": fig_1d}, step=step)
            wandb.log({"probe/slice_1d": wandb.Plotly(fig_1d)}, step=step)

            # --- 2-D Slice (Contour/Surface Plot) ---
            A2 = A[::2]
            Z=[]
            for ai in A2:
                row=[]
                for bj in A2:
                    self._assign_params(theta0 + ai*d1 + bj*d2)
                    row.append(scalar_loss())
                Z.append(row)
            
            # Use Plotly for a 2D contour plot (similar to Matplotlib's heatmap/imshow)
            fig_2d = go.Figure(data=[
                go.Contour(
                    z=np.array(Z), # NumPy is imported at the top now
                    x=A2.tolist(),
                    y=A2.tolist(),
                    colorscale='Jet'
                )
            ])
            fig_2d.update_layout(
                title="2-D Frozen Loss Surface (Contour)",
                xaxis_title="beta",
                yaxis_title="alpha",
                autosize=True,
                height=400,
                width=500
            )

            # Change: Log the Plotly figure directly (replaces wandb.Image)
            # wandb.log({"probe/slice_2d": fig_2d}, step=step)
            wandb.log({"probe/slice_2d": wandb.Plotly(fig_2d)}, step=step)
            # plt.close(fig) # REMOVE this line

    # ----------------- β-frontier -----------------
    @torch.no_grad()
    def _vis_probe_beta_frontier(self, step:int, tag="probe"):
        b = self._to_device_probe()
        betas = list(self._vis_cfg.beta_sweep)
        vals=[]
        old_beta = getattr(self, "beta", 0.0)
        for beta in betas:
            self.beta = float(beta)
            out = self._base_compute_loss(self.model, b)
            vals.append(float(out.detach().cpu()))
        self.beta = old_beta
        
        # Change: Use Plotly Express instead of wandb.plot.line_series
        fig_beta = px.line(x=betas, y=vals, labels={'x':'β', 'y':'Loss'}, 
                           title="β vs Frozen Loss")
        
        # wandb.log({"probe/beta_frontier": fig_beta}, step=step) # Log the Plotly figure
        wandb.log({"probe/beta_frontier": wandb.Plotly(fig_beta)}, step=step)


from trl import GRPOTrainer, GRPOConfig  # or PPOTrainer / DPOTrainer etc.
import wandb

from maths_rewards import gsm8k_numeric_reward
from maths_dataset_loading import load_data, load_test_data
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback

def reward_function(completions, prompts, gold_answer, **kwargs):
    rewards = [gsm8k_numeric_reward(gold_ans, completion) for completion, gold_ans in zip(completions, gold_answer)]
    return rewards

class MyGRPOTrainer(VisualizeTRL, GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vis_setup()
        # optional: ensure W&B is on if you use report_to
        if wandb and wandb.run is None and getattr(self.args, "report_to", None) and "wandb" in self.args.report_to:
            wandb.init(project="rlhf-visuals")


ROOT_DIR="./cached_files"
DIR_FOR_MODEL=f'{ROOT_DIR}/visualisation_models'

seed = 0

model = "Qwen2.5-0.5B"
model = f'{ROOT_DIR}/{model}'
trust_region_model_path = f"{DIR_FOR_MODEL}/{trust_region_method}_{seed}"

run = wandb.init(
    project="visualisations",
    name=f"{trust_region_method}_{seed}",
    reinit=True,
    mode="offline",
    settings=wandb.Settings(symlink=False)  # w/o this get weird saving issues when offline logging on iridis
)

dataset_path = f"{ROOT_DIR}/gsm8k"

tokenizer = AutoTokenizer.from_pretrained(model)

if trust_region_method == "noclip":
    num_iterations = 1
    max_steps = 100    
    trust_region_method = "clip"
else:
    num_iterations = 2
    max_steps = 100

ds = load_data(dataset_path, tokenizer, seed=seed)
train_ds, val_ds = ds["train"], ds["validation"]

if trust_region_method == "clip":
    epsilon = 0.1
    smoothing_alpha = 0
    lr = 1e-6
else:
    epsilon = 0
    smoothing_alpha = 0.1
    lr = 5e-7

# --- 0) your existing config (unchanged) ---
training_args = GRPOConfig(
    # === I/O ===
    output_dir=trust_region_model_path,
    save_steps=100,
    seed=seed,
    num_iterations=num_iterations,
    gradient_checkpointing=False,
    report_to=['wandb'],  # keep as-is; see footnote if you want W&B
    run_name=f"{trust_region_method}_{seed}",
    # num_train_epochs=num_epochs,
    logging_steps=STEP_SIZE,
    max_steps=max_steps,

    # === precision (V100) ===
    fp16=False,
    bf16=True,
    num_generations=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,
    remove_unused_columns=False,
    max_completion_length=128,
    temperature=0.8,
    top_p=0.9,
    beta=0.0,
    # warmup_steps=max(1, int(0.02 * len(train_ds))),
    loss_type='dapo',

    learning_rate=lr,
    epsilon=epsilon,

    # eval
    do_eval=True,
    eval_strategy="steps",
    eval_steps=STEP_SIZE,
    metric_for_best_model="eval_reward",
    load_best_model_at_end=True,

    trust_region_method=trust_region_method,
    smoothing_alpha=smoothing_alpha,
)

# Cadences chosen to line up with your save/eval rhythm (cheap logs every 100 steps; probes every 1000).
training_args.vis_cfg = VisCfg(
    log_every=STEP_SIZE,             # lightweight histograms & scalars
    probe_every=STEP_SIZE,          # heavier frozen-batch probes
    probe_batch_size=8,        # tiny, independent of train batch
    probe_max_tokens=128,      # matches your max_completion_length
    slice_span=0.5,            # ±0.5 * ||θ|| step span for slices
    slice_points=41,           # resolution of the 1-D slice
    beta_sweep=(0.0, 0.01, 0.02, 0.05, 0.1),
    sharp_iters=12,            # keep curvature probe affordable
    lora_only=False,           # set True if you’re training with LoRA and want slices only in adapters
    seed=seed,                 # reuse your experiment seed
)

trainer = MyGRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    log_token_ratios=True,
)

trainer.train()

with open(f"{trust_region_model_path}/token_logs.pkl", 'wb') as f:
    pickle.dump(trainer.token_ratio_logs, f)
