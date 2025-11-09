# analytics_llm_training.py

import os, json, re, math, statistics as stats
import random, itertools
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import pandas as pd
from typing import Dict, Optional
from math import isnan
from re import finditer
from collections import Counter

from itertools import combinations
from plotly.subplots import make_subplots
from statsmodels.stats.proportion import proportion_confint


from maths_rewards import _extract_final_answer, numeric_equal
from reasoning_eval import run_reasoning_analyses
# ---------------------------
# Helpers: parsing & matching
# ---------------------------

MAKE_PLOTS = True

# FINAL_PATTERNS = [
#     re.compile(r"\\boxed\{([-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)\}"),
#     re.compile(r"(?im)^\s*(?:final\s+answer|answer|result)\s*[:\-]\s*([-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?%?)\s*$"),
#     re.compile(r"(?m)^\s*####\s*([-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?)\s*$"),
#     re.compile(r'(?s)"answer"\s*:\s*"?([-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?%?)"?')
# ]

# NUM_TOKEN = re.compile(r"([-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?%?)")

# def _clean_num_token(tok: str):
#     tok = tok.strip()
#     is_pct = tok.endswith("%")
#     if is_pct: tok = tok[:-1]
#     tok = tok.replace(",", "")
#     # fraction?
#     if "/" in tok and tok.count("/") == 1 and all(p.strip() for p in tok.split("/")):
#         try:
#             num = float(tok.split("/")[0]) / float(tok.split("/")[1])
#         except Exception:
#             return None, None
#         return (num * 100.0 if is_pct else num), {"pct": is_pct}
#     # normal float
#     try:
#         num = float(tok)
#         return (num * 100.0 if is_pct else num), {"pct": is_pct}
#     except Exception:
#         return None, None

# def _extract_final_answer(text: str):
#     if not text: 
#         return None
#     # 1) explicit markers
#     for pat in FINAL_PATTERNS:
#         m = pat.search(text)
#         if m:
#             num, _meta = _clean_num_token(m.group(1))
#             if num is not None:
#                 return num
#     # 2) last non-empty line with a standalone number
#     for line in reversed([ln for ln in text.splitlines() if ln.strip()]):
#         # avoid obvious ranges or tuples
#         if re.search(r"\d\s*[\-–—]\s*\d", line) or re.search(r"\(\s*\d", line):
#             continue
#         # choose the ONLY number on the line to avoid ambiguity
#         nums = NUM_TOKEN.findall(line)
#         if len(nums) == 1:
#             num, _meta = _clean_num_token(nums[0])
#             if num is not None:
#                 return num
#     # 3) last numeric token in the whole text (very low confidence)
#     toks = NUM_TOKEN.findall(text)
#     if toks:
#         num, _meta = _clean_num_token(toks[-1])
#         if num is not None:
#             return num
#     return None

# def numeric_equal(pred, gold, atol=1e-6, rtol=1e-6):
#     """Robust numeric comparison with abs and relative tolerances."""
#     try:
#         # normalize to float
#         p = float(pred)
#         g = float(gold)
#     except Exception:
#         return False
#     # exact shortcut
#     if p == g:
#         return True
#     # close?
#     return abs(p - g) <= max(atol, rtol * max(abs(p), abs(g)))

def is_correct(pred, gold, atol=1e-6, rtol=1e-6):
    if gold is None or pred is None:
        return False
    return numeric_equal(pred, gold)


# def is_correct(pred, gold, tol=1e-6):
#     """
#         checks for none before checking if it's numerically equal
#     """
#     if gold is None or pred is None:
#         return False
#     else:
#         return numeric_equal(pred, gold)

def pass_at_k_estimate(c, n, k):
    """OpenAI pass@k estimator: 1 - C(n-c, k) / C(n, k), defined when n >= k."""
    if n < k or k <= 0:
        return np.nan
    if c == 0:
        return 0.0
    # numerically stable product for 1 - comb(n-c, k)/comb(n,k)
    num = 1.0
    for i in range(k):
        num *= (n - c - i) / (n - i)
    return 1.0 - num

def _figure_is_empty(fig):
    # No traces or every trace has no x & y data
    if not fig.data:
        return True
    all_empty = True
    for t in fig.data:
        xs = getattr(t, "x", None)
        ys = getattr(t, "y", None)
        zs = getattr(t, "z", None)
        # Accept heatmaps with z present even if x/y attrs are None
        if (xs is not None and len(xs) > 0) or (ys is not None and len(ys) > 0) or (zs is not None and getattr(np, "size", lambda x: 0)(zs) > 0):
            all_empty = False
            break
    return all_empty

def safe_write_image(fig, path, width=1400, height=900, scale=2):
    # Set explicit size to avoid zero-canvas glitches
    fig.update_layout(width=width, height=height)
    # Skip if empty
    if _figure_is_empty(fig):
        print(f"[skip] empty figure → {path}")
        return
    # Some px density_heatmap figures can have all-NaN z → guard
    try:
        fig.write_image(path + ".png", scale=scale)
        fig.write_html(path + ".html", include_plotlyjs="cdn")

        print(f"[ok] wrote {path}")
    except Exception as e:
        # Fallback: HTML so you still keep the figure
        html_path = os.path.splitext(path)[0] + ".html"
        fig.write_html(html_path, include_plotlyjs="cdn")
        print(f"[fallback] could not write PNG ({e}). Wrote HTML instead → {html_path}")

# ---------------------------
# Loading: multi dataset/model
# ---------------------------

DATASET_NAME_RE = re.compile(r"(?i)^(?P<name>.+?)(?:[_-]raw_completions)?$")  # strip trailing _results or -results

def _infer_model_dataset(root_dir: str, path: Path):
    """
    Expect: <root>/<model_name>/eval_results/<dataset_name>_results.jsonl
    Returns (model, dataset), falling back to 'unknown_*' if not matched.
    """
    model, dataset = "unknown_model", "unknown_dataset"
    try:
        parts = path.relative_to(root_dir).parts  # tuple of str
    except Exception:
        parts = path.parts

    if RESULTS_DIR_TOKEN in parts:
        i = parts.index(RESULTS_DIR_TOKEN)
        # model is the directory before 'eval_results'
        if i >= 1:
            model = parts[i - 1]

        # dataset from filename (strip extension, then strip *_results)
        stem = path.stem  # for .jsonl, this is fine (returns 'dataset_name_results')
        # If you sometimes have double extensions (e.g., .json.gz), use path.name then rsplit once on '.', etc.
        m = DATASET_NAME_RE.match(stem)
        if m:
            dataset = m.group("name")
    return model, dataset

def load_runs(root_dir: str):
    rows = []
    root = Path(root_dir)
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".jsonl"} or RESULTS_DIR_TOKEN not in str(path):
            continue

        model, dataset = _infer_model_dataset(root_dir, path)

        # Read file
        try:
            if path.suffix.lower() == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    records = [json.loads(line) for line in f if line.strip()]
            else:
                with open(path, "r", encoding="utf-8") as f:
                    blob = json.load(f)
                    records = blob if isinstance(blob, list) else [blob]
        except Exception as e:
            print(f"Skipping {path} due to read error: {e}")
            continue

        for r in records:
            temp = float(r.get("temperature", np.nan))
            prompt = r.get("prompt", "")

            # gold extraction (same as before)
            gold = r.get("extracted_gold_answer")
            if gold is None:
                # fallback: parse from gold_answer string
                ga = str(r.get("gold_answer", "")).strip()
                gold = _extract_final_answer(ga)
            gold = None if gold is None else float(gold)

            if gold is None:
                continue

            completions = r.get("completions", []) or []
            rewards = r.get("rewards", []) or [np.nan] * len(completions)
            gen_lens = r.get("gen_lens", []) or [np.nan] * len(completions)
            category = r.get("category", None)
            difficulty = r.get("difficulty", None)

            parsed_answers = [_extract_final_answer(c) for c in completions]

            n = len(completions)
            correct_flags = [is_correct(a, gold) for a in parsed_answers]
            c = sum(correct_flags)
            top1_correct = bool(correct_flags[0]) if n >= 1 else False
            best_of_n_correct = c > 0

            if n > 0:
                keys = [str(a) for a in parsed_answers]
                mv, mv_count = Counter(keys).most_common(1)[0]
                mv_ans = float(mv) if mv not in (None, "None") else None
                majority_vote_correct = is_correct(mv_ans, gold)
                agreement = mv_count / n
            else:
                majority_vote_correct, agreement = False, np.nan

            reward_pick_correct = None
            if n > 0 and any(not pd.isna(x) for x in rewards):
                idx = int(np.nanargmax(np.array(rewards, dtype=float)))
                reward_pick_correct = bool(correct_flags[idx])

            uniq_answers = len({a for a in parsed_answers if a is not None})

            rows.append(dict(
                dataset=dataset, model=model, file=str(path),
                temperature=temp, prompt=prompt,
                gold_answer=gold, completions_n=n,
                completions=completions, parsed_answers=parsed_answers,
                rewards=rewards, gen_lens=gen_lens,
                category=category, difficulty=difficulty,
                top1_correct=top1_correct,
                best_of_n_correct=best_of_n_correct,
                majority_vote_correct=majority_vote_correct,
                reward_pick_correct=reward_pick_correct,
                num_correct=c, uniq_answers=uniq_answers,
                agreement=agreement
            ))

    samples_df = pd.DataFrame(rows)

    # ---- attempt-level explode (unchanged) ----
    attempt_rows = []
    for _, row in samples_df.iterrows():
        n = row["completions_n"]
        for j in range(n):
            attempt_rows.append(dict(
                dataset=row["dataset"], model=row["model"], file=row["file"],
                temperature=row["temperature"], prompt=row["prompt"],
                gold_answer=row["gold_answer"], category=row["category"],
                difficulty=row["difficulty"],
                attempt_idx=j,
                completion=row["completions"][j],
                parsed_answer=row["parsed_answers"][j],
                reward=row["rewards"][j] if j < len(row["rewards"]) else np.nan,
                gen_len=row["gen_lens"][j] if j < len(row["gen_lens"]) else np.nan,
                is_correct=(row["parsed_answers"][j] is not None and row["gold_answer"] is not None and
                            abs(float(row["parsed_answers"][j]) - float(row["gold_answer"])) <= 1e-6)
            ))
    attempts_df = pd.DataFrame(attempt_rows)
    return samples_df, attempts_df

# ---------------------------
# Aggregation: metrics tables
# ---------------------------

def summarize_by(groups, samples_df, attempts_df, passk_list=(1,2,4)):
    """
    Compute metrics grouped by e.g. ["dataset","model","temperature"].
    Returns a DataFrame with one row per group.
    """
    out = []
    for keys, sdf in samples_df.groupby(groups, dropna=False):
        sdf = sdf.copy()
        # collect attempts per group
        mask = (attempts_df[groups] == pd.Series(keys, index=groups)).all(axis=1) if isinstance(keys, tuple) else \
               (attempts_df[groups] == keys)
        adf = attempts_df[mask].copy()

        # base counts
        n_prompts = len(sdf)
        n_atts = len(adf)

        # accuracies
        top1_acc = sdf["top1_correct"].mean() if n_prompts else np.nan
        bestofn_acc = sdf["best_of_n_correct"].mean() if n_prompts else np.nan
        mv_acc = sdf["majority_vote_correct"].mean() if n_prompts else np.nan
        reward_pick_acc = sdf["reward_pick_correct"].dropna().mean() if sdf["reward_pick_correct"].notna().any() else np.nan

        # pass@k (average of per-prompt estimator)
        per_prompt_pass = {k: [] for k in passk_list}
        for _, r in sdf.iterrows():
            c, n = r["num_correct"], r["completions_n"]
            for k in passk_list:
                per_prompt_pass[k].append(pass_at_k_estimate(c, n, min(k, n)))
        passk = {f"pass@{k}": float(np.nanmean(per_prompt_pass[k])) for k in passk_list}

        # reward stats
        reward_vals = adf["reward"].dropna().to_numpy()
        reward_mean = float(np.mean(reward_vals)) if reward_vals.size else np.nan
        reward_std = float(np.std(reward_vals)) if reward_vals.size else np.nan

        # length stats
        gen_vals = adf["gen_len"].dropna().to_numpy()
        gen_mean = float(np.mean(gen_vals)) if gen_vals.size else np.nan

        # diversity / agreement
        uniq_mean = float(sdf["uniq_answers"].replace(0, np.nan).mean())
        agreement_mean = float(sdf["agreement"].mean()) if "agreement" in sdf else np.nan

        # reward ~ correctness correlation (attempt-level)
        corr = np.nan
        if n_atts and adf["reward"].notna().sum() > 5 and adf["is_correct"].nunique() > 1:
            corr = float(np.corrcoef(adf["reward"].fillna(np.nan), adf["is_correct"].astype(int))[0,1])

        row = dict(
            **({g: k for g, k in zip(groups, keys)} if isinstance(keys, tuple) else {groups: keys}),
            n_prompts=n_prompts, n_attempts=n_atts,
            top1_acc=top1_acc, bestofn_acc=bestofn_acc,
            majority_vote_acc=mv_acc, reward_pick_acc=reward_pick_acc,
            reward_mean=reward_mean, reward_std=reward_std,
            gen_len_mean=gen_mean, uniq_answers_mean=uniq_mean,
            agreement_mean=agreement_mean, reward_correct_corr=corr,
            **passk
        )
        out.append(row)

    return pd.DataFrame(out).sort_values(groups if isinstance(groups, list) else [groups]).reset_index(drop=True)

# ---------------------------
# Plotting (Plotly)
# ---------------------------

def plot_accuracy_vs_temperature(summary_df, facet="dataset"):
    fig = px.line(
        summary_df,
        x="temperature",
        y="top1_acc",
        color="model",
        facet_col=facet,
        markers=True,
        hover_data=["bestofn_acc", "majority_vote_acc", "n_prompts"]
    )
    fig.update_layout(title="Top-1 Accuracy vs Temperature", yaxis_title="Top-1 Accuracy")
    return fig

def plot_passk_vs_temperature(summary_df, facet="dataset", ks=("pass@1","pass@2","pass@4")):
    # keep only pass@k cols that exist
    value_vars = [k for k in ks if k in summary_df.columns]
    if not value_vars:
        raise ValueError(f"No pass@k columns found in summary_df. Have: {list(summary_df.columns)}")

    # preserve these columns through melt (only those that exist)
    id_vars = [c for c in ["dataset", "model", "temperature", "n_prompts"] if c in summary_df.columns]

    melted = summary_df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="metric",
        value_name="value"
    )

    # only include hover fields that are present post-melt
    hover_cols = [c for c in ["n_prompts"] if c in melted.columns]

    fig = px.line(
        melted,
        x="temperature",
        y="value",
        color="model",
        line_dash="metric",
        facet_col=facet,
        markers=True,
        hover_data=hover_cols
    )
    fig.update_layout(title="pass@k vs Temperature", yaxis_title="pass@k")
    return fig

def plot_reward_distributions(attempts_df, facet="dataset"):
    fig = px.violin(
        attempts_df, x="temperature", y="reward", color="model",
        box=True, points="all", facet_col=facet
    )
    fig.update_layout(title="Reward Distribution by Temperature")
    return fig

def plot_diversity_vs_temp(summary_df, facet="dataset"):
    fig = px.line(
        summary_df, x="temperature", y="uniq_answers_mean",
        color="model", markers=True, facet_col=facet,
        hover_data=["agreement_mean"]
    )
    fig.update_layout(title="Answer Diversity vs Temperature", yaxis_title="Mean # Unique Answers")
    return fig

def plot_len_vs_reward(attempts_df, facet="dataset"):
    fig = px.scatter(
        attempts_df, x="gen_len", y="reward", color="model",
        symbol="temperature", facet_col=facet, trendline="ols"
    )
    fig.update_layout(title="Generation Length vs Reward", xaxis_title="gen_len (tokens/chars)")
    return fig

def plot_category_heatmap_per_dataset(
    samples_df,
    facet_wrap=3,
    show_values=True,
    mark_best=True,
    fmt="{:.2f}",
    force_overlay_text=False,   # set True if your env hides Heatmap text
    set_max=True,
):
    """
    Creates one heatmap figure per dataset:
      y = category
      x = temperature (categorical labels of temperatures present)
      z = mean top-1 accuracy
    Faceted by model, wrapping every `facet_wrap` columns.

    Returns: dict {dataset_name: plotly_figure}
    """
    figs = {}
    df = samples_df.copy()

    # ---- validation ----
    if "top1_correct" not in df.columns:
        raise ValueError("`samples_df` must contain 'top1_correct' as boolean per prompt.")
    if "category" not in df.columns:
        raise ValueError("`samples_df` must contain a 'category' column.")

    df["category"] = df["category"].fillna("uncategorized")
    df = df[df["top1_correct"].notna()]

    # aggregate mean accuracy
    agg = (
        df.groupby(["dataset", "model", "temperature", "category"], dropna=False)["top1_correct"]
          .mean()
          .reset_index()
    )

    for ds, sdf in agg.groupby("dataset", dropna=False):
        sdf = sdf.dropna(subset=["temperature", "category", "top1_correct"]).copy()
        if sdf.empty:
            continue

        # categorical x labels for temperatures present
        temps_present = sorted(sdf["temperature"].dropna().unique().tolist())
        if not temps_present:
            continue
        sdf["temp_label"] = sdf["temperature"].map(lambda t: f"{t:g}")
        temp_labels = [f"{t:g}" for t in temps_present]

        # order categories (alphabetical; customize as needed)
        cat_order = sorted(sdf["category"].unique().tolist())

        models = [m for m in sorted(sdf["model"].dropna().unique().tolist())]
        if not models:
            continue

        # subplot grid
        n = len(models)
        cols = min(facet_wrap, n)
        rows = math.ceil(n / cols)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[str(m) for m in models],
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
            shared_yaxes=True,
        )

        # add one Heatmap per model
        for k, model in enumerate(models):
            r = (k // cols) + 1
            c = (k % cols) + 1
            ms = sdf[sdf["model"] == model]

            # pivot to 2D grid
            pivot = (
                ms.pivot_table(index="category", columns="temp_label", values="top1_correct", aggfunc="mean")
                  .reindex(index=cat_order, columns=temp_labels)
            )
            Z = pivot.values.astype(float)

            text = None
            if show_values and not force_overlay_text:
                text = [[(fmt.format(v) if np.isfinite(v) else "") for v in row] for row in Z]

            heatmap = go.Heatmap(
                z=Z,
                x=pivot.columns,
                y=pivot.index,
                coloraxis="coloraxis",
                text=text,
                texttemplate="%{text}" if text is not None else None,
                textfont={"size": 11, "color": "white"},
                hovertemplate="Temperature=%{x}<br>Category=%{y}<br>Top-1 Acc=%{z:.1%}<extra></extra>",
                hoverongaps=False,
            )
            fig.add_trace(heatmap, row=r, col=c)

            # optional overlay text (rock-solid fallback)
            if show_values and (force_overlay_text or text is None):
                x_vals = list(pivot.columns)
                y_vals = list(pivot.index)
                overlay_x, overlay_y, overlay_text = [], [], []
                for i, ylab in enumerate(y_vals):
                    for j, xlab in enumerate(x_vals):
                        v = Z[i, j]
                        overlay_x.append(xlab)
                        overlay_y.append(ylab)
                        overlay_text.append(fmt.format(v) if np.isfinite(v) else "")

                # shadow layer (black) for contrast
                fig.add_trace(
                    go.Scatter(
                        x=overlay_x, y=overlay_y, mode="text", text=overlay_text,
                        textposition="middle center", textfont=dict(size=13, color="black"),
                        hoverinfo="skip", showlegend=False
                    ),
                    row=r, col=c
                )
                # main white text
                fig.add_trace(
                    go.Scatter(
                        x=overlay_x, y=overlay_y, mode="text", text=overlay_text,
                        textposition="middle center", textfont=dict(size=11, color="white"),
                        hoverinfo="skip", showlegend=False
                    ),
                    row=r, col=c
                )

            # mark best cell with a star
            if mark_best and np.isfinite(Z).any():
                with np.errstate(invalid="ignore"):
                    i, j = np.unravel_index(np.nanargmax(Z), Z.shape)
                fig.add_annotation(
                    x=pivot.columns[j],
                    y=pivot.index[i],
                    text="⭐",
                    showarrow=False,
                    font={"size": 16},
                    row=r, col=c
                )

        colour_axis = {"colorscale": "Viridis", "cmin": 0}
        if set_max:
            coloraxis["cmax"] = 1
        # shared styling
        fig.update_layout(
            title=f"Top-1 Accuracy by Category × Temperature — {ds}",
            width=1400,
            height=300 + 300 * rows,
            coloraxis=colour_axis,
            coloraxis_colorbar_title="Top-1 Acc",
            uniformtext_minsize=10,
            uniformtext_mode="show",
        )

        # treat axes as categorical with explicit order
        for i in range(1, rows * cols + 1):
            xk = f"xaxis{i}" if i > 1 else "xaxis"
            yk = f"yaxis{i}" if i > 1 else "yaxis"
            if xk in fig.layout:
                fig.layout[xk].type = "category"
                fig.layout[xk].categoryorder = "array"
                fig.layout[xk].categoryarray = temp_labels
                fig.layout[xk].title = "Temperature"
            if yk in fig.layout:
                fig.layout[yk].categoryorder = "array"
                fig.layout[yk].categoryarray = cat_order
                fig.layout[yk].title = "Category"

        figs[ds] = fig

    return figs

# --------------------------
# Utilities
# --------------------------

def mask_models(names, seed=0):
    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)
    mapping = {orig: f"Model {chr(ord('A')+i)}" for i, orig in enumerate(shuffled)}
    return mapping

def choose_stratified_samples(df, n_per_category=10, category_col="category", seed=0):
    rng = np.random.default_rng(seed)
    picks = []
    for cat, g in df.groupby(category_col, dropna=False):
        take = min(n_per_category, len(g))
        picks.append(g.sample(take, random_state=seed))
    return pd.concat(picks, ignore_index=True) if picks else df.sample(min(50, len(df)), random_state=seed)

# --------------------------
# 1) Likert rubric sheet
# --------------------------

def make_likert_sheet(samples_df, models=None, datasets=None, temperatures=None,
                      n_per_category=10, category_col="category", hide_gold=False, seed=0):
    """
    Produces a table for human scoring with Likert fields.
    One row = (prompt, model, temperature). Uses the *first* completion text.
    """
    df = samples_df.copy()

    if models:       df = df[df["model"].isin(models)]
    if datasets:     df = df[df["dataset"].isin(datasets)]
    if temperatures: df = df[df["temperature"].isin(temperatures)]

    # pick one completion to display (top-1); you can switch to majority-vote text if you store it
    df["output_text"] = df["completions"].map(lambda xs: xs[0] if isinstance(xs, list) and xs else "")

    # stratify by category to keep the review balanced
    base = choose_stratified_samples(df, n_per_category=n_per_category, category_col=category_col, seed=seed)

    # mask model identities
    mapping = mask_models(sorted(base["model"].unique().tolist()), seed=seed)
    base = base.assign(model_masked=base["model"].map(mapping))

    cols = [
        "dataset","model_masked","temperature","category","difficulty","prompt","output_text"
    ]
    if not hide_gold:
        cols += ["gold_answer"]

    sheet = base[cols].copy()
    # add empty Likert columns to be filled by annotators
    for c in ["correctness_1to5","reasoning_1to5","faithfulness_1to5","format_1to5","notes"]:
        sheet[c] = ""

    return sheet, mapping

# --------------------------
# 2) Pairwise Arena sheet
# --------------------------

def make_arena_sheet(samples_df, models=None, datasets=None, temperatures=None,
                     pairs_per_prompt=1, n_prompts=100, seed=0, hide_gold=True):
    """
    Builds a blinded pairwise sheet: for each prompt, sample 2 models (same temperature),
    provide outputs as A/B, and leave 'winner' to be filled (A/B/Tie/Skip).
    """
    rng = np.random.default_rng(seed)
    df = samples_df.copy()

    if datasets:     df = df[df["dataset"].isin(datasets)]
    if temperatures: df = df[df["temperature"].isin(temperatures)]
    if models:       df = df[df["model"].isin(models)]

    # only prompts that appear for at least 2 models at the same temperature
    key_cols = ["dataset","prompt","temperature"]
    counts = df.groupby(key_cols)["model"].nunique().reset_index(name="n_models")
    valid_keys = counts[counts["n_models"] >= 2][key_cols]
    df = df.merge(valid_keys, on=key_cols, how="inner")

    # First completion text:
    df["output_text"] = df["completions"].map(lambda xs: xs[0] if isinstance(xs, list) and xs else "")

    # sample prompt keys
    unique_keys = df[key_cols].drop_duplicates()
    if len(unique_keys) == 0:
        raise ValueError("No overlapping prompts across models at the same temperature.")
    take = min(n_prompts, len(unique_keys))
    sampled_keys = unique_keys.sample(take, random_state=seed)

    rows = []
    for _, k in sampled_keys.iterrows():
        sub = df[(df["dataset"]==k["dataset"]) &
                 (df["prompt"]==k["prompt"]) &
                 (df["temperature"]==k["temperature"])]
        ms = sorted(sub["model"].unique().tolist())
        if len(ms) < 2:
            continue

        # build up to pairs_per_prompt pairs without replacement when possible
        all_pairs = list(itertools.combinations(ms, 2))
        rng.shuffle(all_pairs)
        for (m1, m2) in all_pairs[:pairs_per_prompt]:
            o1 = sub[sub["model"]==m1].iloc[0]
            o2 = sub[sub["model"]==m2].iloc[0]
            # blind assignment A/B
            if rng.random() < 0.5:
                A, B = o1, o2
            else:
                A, B = o2, o1

            row = dict(
                dataset=k["dataset"],
                temperature=k["temperature"],
                prompt=k["prompt"],
                model_A=A["model"], model_B=B["model"],
                output_A=A["output_text"], output_B=B["output_text"],
            )
            if not hide_gold:
                row["gold_answer"] = A["gold_answer"]  # same for both
            row["winner"] = ""  # to be filled: A/B/Tie/Skip
            row["notes"] = ""
            rows.append(row)

    return pd.DataFrame(rows)

# --------------------------
# 3) Analyze Arena preferences
# --------------------------

def analyze_arena(arena_csv):
    """
    Reads an arena CSV with columns: model_A, model_B, winner in {A,B,Tie,Skip}.
    Returns win rates and Bradley–Terry scores.
    """
    df = pd.read_csv(arena_csv)
    df = df[df["winner"].isin(["A","B","Tie"])].copy()

    # win matrix
    models = sorted(set(df["model_A"]) | set(df["model_B"]))
    idx = {m:i for i,m in enumerate(models)}
    wins = np.zeros((len(models), len(models)), dtype=float)
    comps = np.zeros_like(wins)

    for _, r in df.iterrows():
        i, j = idx[r["model_A"]], idx[r["model_B"]]
        comps[i,j] += 1; comps[j,i] += 1
        if r["winner"] == "A":
            wins[i,j] += 1
        elif r["winner"] == "B":
            wins[j,i] += 1
        elif r["winner"] == "Tie":
            wins[i,j] += 0.5; wins[j,i] += 0.5

    # win rates
    winrate = pd.DataFrame(index=models, columns=models, dtype=float)
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if comps[i,j] > 0:
                winrate.loc[mi, mj] = wins[i,j] / comps[i,j]
            else:
                winrate.loc[mi, mj] = np.nan

    # Bradley–Terry via minorize–maximize (simple)
    # ability s_m; P(m beats n) = exp(s_m) / (exp(s_m)+exp(s_n))
    s = np.zeros(len(models))
    for _ in range(200):
        s_old = s.copy()
        for m in range(len(models)):
            num = 0.0; den = 0.0
            for n in range(len(models)):
                if m==n: continue
                w_mn = wins[m,n]     # wins of m over n
                c_mn = comps[m,n]    # total comparisons
                if c_mn == 0: continue
                num += w_mn
                den += c_mn / (1 + math.exp(s[n] - s[m]))
            if den > 0:
                s[m] = math.log(max(num, 1e-12) / den)
        if np.linalg.norm(s - s_old) < 1e-6:
            break

    scores = pd.DataFrame({
        "model": models,
        "bt_score": s - np.mean(s)  # centered
    }).sort_values("bt_score", ascending=False).reset_index(drop=True)

    return winrate, scores

# --------------------------
# Convenience: write sheets
# --------------------------

def export_likert_and_arena(samples_df, out_dir="qual_sheets",
                            likert_kwargs=None, arena_kwargs=None, seed=0):
    os.makedirs(out_dir, exist_ok=True)

    sheet, mapping = make_likert_sheet(samples_df, seed=seed, **(likert_kwargs or {}))
    # Save masked mapping separately for later unmasking
    mask_df = pd.DataFrame(sorted(mapping.items()), columns=["model_real","model_masked"])
    sheet.to_csv(Path(out_dir) / "likert_sheet.csv", index=False)
    mask_df.to_csv(Path(out_dir) / "likert_mask_mapping.csv", index=False)

    arena = make_arena_sheet(samples_df, seed=seed, **(arena_kwargs or {}))
    arena.to_csv(Path(out_dir) / "arena_sheet.csv", index=False)

    print(f"Wrote:\n- {out_dir}/likert_sheet.csv\n- {out_dir}/likert_mask_mapping.csv\n- {out_dir}/arena_sheet.csv")

def plot_difficulty_heatmap_per_dataset(samples_df, facet_wrap=3, show_values=True, mark_best=True, fmt="{:.2f}", set_max=True):
    """
    Creates one heatmap per dataset:
      y = difficulty
      x = temperature (categorical labels of temperatures present)
      z = mean top-1 accuracy
    Faceted by model and wrapped every `facet_wrap` columns.

    Returns: dict {dataset_name: plotly_figure}
    """
    figs = {}
    df = samples_df.copy()

    # ---- validation ----
    if "top1_correct" not in df.columns:
        raise ValueError("`samples_df` must contain 'top1_correct' as boolean per prompt.")
    if "difficulty" not in df.columns:
        raise ValueError("`samples_df` must contain 'difficulty'.")

    df["difficulty"] = df["difficulty"].astype(str).fillna("unknown")
    df = df[df["top1_correct"].notna()]

    # aggregate mean accuracy
    agg = (
        df.groupby(["dataset", "model", "temperature", "difficulty"], dropna=False)["top1_correct"]
          .mean()
          .reset_index()
    )

    for ds, sdf in agg.groupby("dataset", dropna=False):
        sdf = sdf.dropna(subset=["temperature", "difficulty", "top1_correct"]).copy()
        if sdf.empty:
            continue

        # categorical x labels for temperatures present
        temps_present = sorted(sdf["temperature"].dropna().unique().tolist())
        if not temps_present:
            continue
        sdf["temp_label"] = sdf["temperature"].map(lambda t: f"{t:g}")
        temp_labels = [f"{t:g}" for t in temps_present]

        # order difficulty labels
        def _maybe_float(x):
            try:
                return float(x)
            except Exception:
                return None
        diffs = sdf["difficulty"].unique().tolist()
        diffs_num = [_maybe_float(d) for d in diffs]
        if all(v is not None for v in diffs_num):
            diff_order = [d for _, d in sorted(zip(diffs_num, diffs), key=lambda z: z[0])]
        else:
            common = ["easy", "medium", "hard", "very hard"]
            lower_map = {d.lower(): d for d in diffs}
            if set(lower_map.keys()).issuperset(set(common)):
                diff_order = [lower_map[k] for k in common if k in lower_map]
                extras = sorted([d for d in diffs if d not in diff_order])
                diff_order += extras
            else:
                diff_order = sorted(diffs)

        models = [m for m in sorted(sdf["model"].dropna().unique().tolist())]
        if not models:
            continue

        # subplot grid
        n = len(models)
        cols = min(facet_wrap, n)
        rows = math.ceil(n / cols)

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[str(m) for m in models],
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
            shared_yaxes=True,
        )

        # helper to format axis key names
        def _axis_key(i, which="x"):
            return f"{which}axis" if i == 1 else f"{which}axis{i}"

        # add one Heatmap per model
        for k, model in enumerate(models):
            r = (k // cols) + 1
            c = (k % cols) + 1
            ms = sdf[sdf["model"] == model]

            # pivot to 2D grid
            pivot = (
                ms.pivot_table(index="difficulty", columns="temp_label", values="top1_correct", aggfunc="mean")
                  .reindex(index=diff_order, columns=temp_labels)
            )
            Z = pivot.values.astype(float)

            text = None
            if show_values:
                text = [[(fmt.format(v) if np.isfinite(v) else "") for v in row] for row in Z]

            fig.add_trace(
                go.Heatmap(
                    z=Z,
                    x=pivot.columns,
                    y=pivot.index,
                    coloraxis="coloraxis",
                    text=text,
                    texttemplate="%{text}",
                    textfont={"size": 11},
                    hovertemplate="Temperature=%{x}<br>Difficulty=%{y}<br>Top-1 Acc=%{z:.1%}<extra></extra>",
                ),
                row=r, col=c
            )

            # mark best cell with a star
            if mark_best and np.isfinite(Z).any():
                with np.errstate(invalid="ignore"):
                    i, j = np.unravel_index(np.nanargmax(Z), Z.shape)
                fig.add_annotation(
                    x=pivot.columns[j],
                    y=pivot.index[i],
                    text="⭐",
                    showarrow=False,
                    font={"size": 16},
                    row=r, col=c
                )

        colour_axis = {"colorscale": "Viridis", "cmin": 0}
        if set_max:
            coloraxis["cmax"] = 1

        # shared styling
        fig.update_layout(
            title=f"Top-1 Accuracy by Difficulty × Temperature — {ds}",
            width=1400,
            height=300 + 300 * rows,  # scale a bit with rows
            coloraxis=colour_axis,
        )

        # axis categories and titles
        for i in range(1, rows * cols + 1):
            xk = _axis_key(i, "x")
            yk = _axis_key(i, "y")
            if xk in fig.layout:
                fig.layout[xk].type = "category"
                fig.layout[xk].categoryorder = "array"
                fig.layout[xk].categoryarray = temp_labels
            if yk in fig.layout:
                fig.layout[yk].categoryorder = "array"
                fig.layout[yk].categoryarray = diff_order

        # add axis titles on outer edges
        for c in range(1, cols + 1):
            fig.update_xaxes(title_text="Temperature", row=rows, col=c)
        for r in range(1, rows + 1):
            fig.update_yaxes(title_text="Difficulty", row=r, col=1)

        # Ensure heatmaps actually render their text labels
        fig.update_traces(
            selector=dict(type="heatmap"),
            texttemplate="%{text}",            # tell Plotly to use the text array
            hoverongaps=False,                 # don’t suppress hover/text on NaN-adjacent cells
            textfont=dict(size=11, color="white")  # white reads well on Viridis
        )
        fig.update_layout(
            uniformtext_minsize=10,
            uniformtext_mode="show"
        )

        figs[ds] = fig

    return figs

# difference heatmaps
def _agg_grid(samples_df, axis="category"):
    assert axis in {"category","difficulty"}
    df = samples_df.copy()
    df[axis] = df[axis].astype(str).fillna("unknown")
    df = df[df["top1_correct"].notna()]
    g = (df.groupby(["dataset","model","temperature",axis], dropna=False)["top1_correct"]
           .mean()
           .reset_index())
    g["temp_label"] = g["temperature"].map(lambda t: f"{t:g}")
    return g

def _pivot_for(ds_df, model, axis, x_order=None, y_order=None):
    sub = ds_df[ds_df["model"] == model].copy()
    if sub.empty:
        return None
    if x_order is None:
        x_order = sorted(sub["temp_label"].unique(), key=lambda s: float(s))
    if y_order is None:
        y_order = sorted(sub[axis].unique())
    piv = (sub.pivot_table(index=axis, columns="temp_label", values="top1_correct", aggfunc="mean")
               .reindex(index=y_order, columns=x_order))
    return piv

def diff_heatmaps_per_dataset(samples_df, axis="category", baseline_model=None,
                              facet_wrap=3, symmetric_range=0.25):
    """
    Build per-dataset heatmaps of accuracy differences:
      - If baseline_model is given: each subplot is (model − baseline).
      - Else: all pairwise (A − B).
    Cells are annotated with the diff rounded to 2 d.p.
    """
    ds_figs = {}
    agg = _agg_grid(samples_df, axis=axis)

    for ds, ds_df in agg.groupby("dataset", dropna=False):
        models = sorted(ds_df["model"].unique().tolist())
        if baseline_model and baseline_model not in models:
            continue

        pairs = ([(m, baseline_model) for m in models if m != baseline_model]
                 if baseline_model else list(combinations(models, 2)))
        if not pairs:
            continue

        titles = [f"{a} − {b}" for (a, b) in pairs]
        ncols = min(facet_wrap, len(pairs))
        nrows = int(np.ceil(len(pairs) / ncols))
        fig = make_subplots(rows=nrows, cols=ncols,
                            subplot_titles=titles,
                            shared_xaxes=False, 
                            shared_yaxes=True)

        # consistent axes across subplots
        all_x = sorted(ds_df["temp_label"].unique(), key=lambda s: float(s))
        all_y = sorted(ds_df[axis].astype(str).unique())

        for k, (mA, mB) in enumerate(pairs, start=1):
            r = int(np.ceil(k / ncols))
            c = ((k - 1) % ncols) + 1

            pivA = _pivot_for(ds_df, mA, axis, x_order=all_x, y_order=all_y)
            pivB = _pivot_for(ds_df, mB, axis, x_order=all_x, y_order=all_y)
            if pivA is None or pivB is None:
                continue

            diff = pivA - pivB
            z = diff.values
            text = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.2f}")(z))

            heat = go.Heatmap(
                z=z,
                x=list(diff.columns),
                y=list(diff.index),
                colorscale="RdBu",
                zmid=0.0,
                zmin=-symmetric_range, zmax=symmetric_range,
                showscale=(k == len(pairs)),  # one colorbar on the last subplot
                colorbar=dict(title="Δ Acc")
            )
            # put labels INSIDE the trace
            heat.update(text=text, texttemplate="%{text}", textfont=dict(color="black", size=11))

            fig.add_trace(heat, row=r, col=c)

        fig.update_layout(
            title=f"Accuracy Difference Heatmaps ({axis.title()} × Temperature) — {ds}",
            xaxis_title="Temperature", yaxis_title=axis.title(),
            width=max(900, 500*ncols), height=max(600, 420*nrows),
            margin=dict(t=80, r=20, b=40, l=80)
        )

        # Make all x-axes categorical (temperature labels)
        for ax_name, _ in fig.to_plotly_json()["layout"].items():
            if str(ax_name).startswith("xaxis"):
                fig.layout[ax_name].type = "category"

        ds_figs[ds] = fig

    return ds_figs

def publication_ready_summary(df, gsm8k_model="baseline"):
    """
    Summarize evaluation results from your summary dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Your summary dataframe with columns like:
        [dataset, model, temperature, top1_acc, bestofn_acc, majority_vote_acc, n_prompts]
    gsm8k_model : str
        Which model to treat as the GSM8K in-domain baseline for generalization gaps.

    Returns
    -------
    summary : pd.DataFrame
        Table with best top-1, best SC@10, gain, generalization gap, and 95% CIs.
    """

    # pick SC metric (best-of-n preferred, fallback to majority vote)
    df = df.copy()
    if "bestofn_acc" in df.columns:
        df["sc_acc"] = df["bestofn_acc"]
    elif "majority_vote_acc" in df.columns:
        df["sc_acc"] = df["majority_vote_acc"]
    else:
        raise ValueError("Need either bestofn_acc or majority_vote_acc in dataframe.")

    rows = []

    for (dataset, model), g in df.groupby(["dataset", "model"]):
        # best top-1 and SC across temperatures
        top1_row = g.loc[g["top1_acc"].idxmax()]
        sc_row = g.loc[g["sc_acc"].idxmax()]

        top1 = top1_row["top1_acc"]
        sc = sc_row["sc_acc"]
        n = int(top1_row["n_prompts"])  # assume same across temps

        # Wilson CI for top-1
        ci_low, ci_high = proportion_confint(count=int(top1 * n), nobs=n, method="wilson")
        ci_text = f"[{ci_low:.2f}, {ci_high:.2f}]"

        rows.append({
            "dataset": dataset,
            "model": model,
            "best_top1": round(top1, 3),
            "top1_CI95": ci_text,
            "best_temp_top1": top1_row["temperature"],
            "best_sc": round(sc, 3),
            "best_temp_sc": sc_row["temperature"],
            "gain": round(sc - top1, 3),
            "n_prompts": n,
        })

    summary = pd.DataFrame(rows)

    # attach generalization gap relative to GSM8K (choose a reference model)
    gsm8k_acc = summary.query("dataset == 'gsm8k' and model == @gsm8k_model")["best_top1"].values
    if len(gsm8k_acc) > 0:
        gsm8k_acc = gsm8k_acc[0]
        summary["gen_gap_vs_gsm8k"] = (summary["best_top1"] - gsm8k_acc).round(3)
        summary["rel_to_gsm8k"] = (summary["best_top1"] / gsm8k_acc).round(2)
    else:
        summary["gen_gap_vs_gsm8k"] = np.nan
        summary["rel_to_gsm8k"] = np.nan

    return summary

from collections import defaultdict

def randomly_select_prompts(attempts_df, n=5, dataset=None, temperature=None, blind_models=False):
    """
    Find most polarising prompts across models for a given dataset/temperature.

    A prompt is 'polarising' if some models get it right and others wrong.

    Parameters
    ----------
    attempts_df : pd.DataFrame
        Output of load_runs() (attempts_df).
    n : int
        Number of prompts to return.
    dataset : str or None
        Filter to a single dataset (optional).
    temperature : float or None
        Filter to a single temperature (optional).

    Returns
    -------
    polarising_df : pd.DataFrame
        Subset of attempts_df for the selected prompts.
    """
    print(f"Randomly selecting {n} prompts and printing a random response per model for the {dataset} dataset.")

    df = attempts_df.copy()
    if dataset is not None:
        df = df[df["dataset"] == dataset]
    if temperature is not None:
        df = df[df["temperature"] == temperature]

    # 1) build a summary to find prompts with >1 model
    summary = (
        df.groupby("prompt")["model"].nunique().reset_index(name="n_models")
    )
    eligible_prompts = summary.loc[summary["n_models"] > 1, "prompt"]

    # 2) sample prompts
    random_prompts = eligible_prompts.sample(n=n, random_state=42)

    # 3) iterate prompts, then subset ORIGINAL df and group by model
    for prompt in random_prompts:
        print("=" * 100)
        print("PROMPT:")
        print(str(prompt).strip())
        print()

        sub = df[df["prompt"] == prompt]

        for model, g2 in sub.groupby("model"):
            chosen = g2.sample(1, random_state=42).iloc[0]  # pick one row
            comp = str(chosen["completion"]).strip().replace("\n", " ")
            print(f"--- {model}")
            print(comp)
            print()

    return random_prompts

def find_polarising_prompts(attempts_df, n=5, dataset=None, temperature=None):
    """
    Find most polarising prompts across models for a given dataset/temperature.

    A prompt is 'polarising' if some models get it right and others wrong.

    Parameters
    ----------
    attempts_df : pd.DataFrame
        Output of load_runs() (attempts_df).
    n : int
        Number of prompts to return.
    dataset : str or None
        Filter to a single dataset (optional).
    temperature : float or None
        Filter to a single temperature (optional).

    Returns
    -------
    polarising_df : pd.DataFrame
        Subset of attempts_df for the selected prompts.
    """
    print(f"Finding top {n} most ~polarising prompts and printing the 'best' response per model for the {dataset} dataset.")

    df = attempts_df.copy()
    if dataset is not None:
        df = df[df["dataset"] == dataset]
    if temperature is not None:
        df = df[df["temperature"] == temperature]

    # group by prompt, measure std of is_correct across models
    grouped = df.groupby("prompt").agg(
        mean_acc=("is_correct", "mean"),
        std_acc=("is_correct", "std"),
        n_models=("model", "nunique"),
    ).reset_index()

    # need >1 model to compare
    grouped = grouped[grouped.n_models > 1]

    # top-n by std (polarisation)
    top_prompts = grouped.sort_values("std_acc", ascending=False).head(n)["prompt"]

    return df[df["prompt"].isin(top_prompts)].sort_values(["prompt", "model", "attempt_idx"])

def print_polarising_responses(polarising_df, max_chars=400):
    """
    Print per-prompt, per-model completions for the most polarising prompts.

    Shows the 'best' completion (first correct if any; otherwise first).
    """
    for prompt, g in polarising_df.groupby("prompt"):
        print("=" * 100)
        print("PROMPT:")
        print(prompt.strip())
        print()

        for model, g2 in g.groupby("model"):
            # choose best completion: any correct? else first
            correct_rows = g2[g2["is_correct"] == True]
            if not correct_rows.empty:
                chosen = correct_rows.iloc[0]  # first correct completion
                mark = "✅"
            else:
                chosen = g2.iloc[0]  # fallback to first attempt
                mark = "❌"

            comp = chosen["completion"].strip().replace("\n", " ")
            print(f"--- {model} {mark}")
            print(comp[:max_chars])
            print()

import re
import numpy as np
import pandas as pd
import plotly.express as px
from statsmodels.stats.proportion import proportion_confint

# ------------------------
# Helpers
# ------------------------

def split_model_seed(m):
    """
    Split 'name_seed' → ('name', 'seed') where seed is int if present, else None.
    Examples: 'smoothing_ratio_0' -> ('smoothing_ratio', 0), 'default_12' -> ('default', 12),
              'baseline_0' -> ('baseline', 0), 'qwen2.5-0.5b' -> ('qwen2.5-0.5b', None)
    """
    m = str(m)
    m = m.strip()
    m = m.replace(" ", "")
    m = m.replace("-", "_") if False else m  # keep hyphens if you like
    m = m.strip("_")
    m = m
    # look for _<digits> at the end
    m = str(m)
    mat = re.search(r"^(.*)_(\d+)$", m)
    if mat:
        base = mat.group(1)
        seed = int(mat.group(2))
        return base, seed
    return m, None


def _pooled_binom(series_acc, series_n):
    """
    Pooled accuracy across seeds: sum(correct)/sum(n)
    Returns (acc, n_correct, n_total, ci_low, ci_high)
    """
    valid = (~series_acc.isna()) & (~series_n.isna()) & (series_n > 0)
    if not valid.any():
        return np.nan, 0, 0, np.nan, np.nan
    n_total = int(series_n[valid].sum())
    n_correct = int(np.round((series_acc[valid] * series_n[valid]).sum()))
    acc = n_correct / n_total if n_total > 0 else np.nan
    if n_total > 0:
        ci_low, ci_high = proportion_confint(count=n_correct, nobs=n_total, method="wilson")
    else:
        ci_low, ci_high = (np.nan, np.nan)
    return acc, n_correct, n_total, ci_low, ci_high


def aggregate_over_seeds(summary_df, metric_cols=None, n_col="n_prompts"):
    """
    Collapse {model_seed} into model families per (dataset, model_base, temperature).
    Computes pooled accuracy + Wilson 95% CI per metric, and per-seed std (optional).

    Parameters
    ----------
    summary_df : DataFrame with columns:
        ['dataset','model','temperature', n_col, ... metrics ...]
    metric_cols : list of metric column names to aggregate. If None, will auto-detect:
        ['top1_acc', 'bestofn_acc', 'majority_vote_acc', 'pass@1', 'pass@2', 'pass@4'] if present.
    n_col : str name of the denominator column (use n_scorable if you have it)

    Returns
    -------
    agg_df : long-form DataFrame with columns:
        ['dataset','model_base','temperature','metric','mean','n_correct','n_total','ci_low','ci_high','seed_count','std_across_seeds']
    """
    df = summary_df.copy()

    # pick denominator column
    if "n_scorable" in df.columns:
        n_col = "n_scorable"
    else:
        n_col = n_col

    # auto-detect metrics
    default_candidates = ["top1_acc", "bestofn_acc", "majority_vote_acc", "pass@1", "pass@2", "pass@4", "pass@8", "pass@16", "gen_len_mean"]
    if metric_cols is None:
        metric_cols = [c for c in default_candidates if c in df.columns]
        if not metric_cols:
            raise ValueError("No metric columns found. Please pass metric_cols explicitly.")

    # add model_base and seed
    out = df.copy()
    out[["model_base","seed"]] = out["model"].apply(lambda m: pd.Series(split_model_seed(m)))

    rows = []
    group_keys = ["dataset", "model_base", "temperature"]

    for (dataset, model_base, temp), g in out.groupby(group_keys):
        seed_count = g["seed"].nunique() if g["seed"].notna().any() else g["model"].nunique()

        for metric in metric_cols:
            if metric not in g.columns:
                continue

            acc, n_correct, n_total, ci_low, ci_high = _pooled_binom(g[metric], g[n_col])
            # std across seeds (treating each seed's accuracy)
            # Only compute if we can; otherwise NaN
            # Note: this is std of per-seed means, NOT binomial std.
            std_seed = g.groupby("seed")[metric].mean().std(ddof=1) if g["seed"].notna().any() else g[metric].std(ddof=1)

            rows.append({
                "dataset": dataset,
                "model_base": model_base,
                "temperature": temp,
                "metric": metric,
                "mean": acc,
                "n_correct": n_correct,
                "n_total": n_total,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci95": (ci_high - acc) if (np.isfinite(ci_high) and not np.isnan(acc)) else np.nan,  # upper error
                "seed_count": seed_count,
                "std_across_seeds": std_seed,
            })

    agg_df = pd.DataFrame(rows)
    return agg_df


# ------------------------
# Plotting
# ------------------------

def plot_metric_vs_temperature_avg(agg_df, metric="top1_acc", facet="dataset", title=None, use_ci=True, use_std=False):
    """
    Plot a single metric vs temperature, averaged over seeds, with error bars.
    - use_ci: show 95% CI (pooled binomial)
    - use_std: show per-seed std as error bars (ignored if use_ci=True)
    """
    data = agg_df[agg_df["metric"] == metric].copy()
    if data.empty:
        raise ValueError(f"No rows for metric={metric}")

    # choose error column
    if use_ci:
        err_col = "ci95"
    elif use_std:
        err_col = "std_across_seeds"
    else:
        err_col = None

    fig = px.line(
        data.sort_values(["dataset","model_base","temperature"]),
        x="temperature", y="mean",
        color="model_base",
        facet_col=facet, facet_col_wrap=2 if data[facet].nunique() > 2 else 0,
        markers=True,
        error_y=err_col if err_col else None,
        hover_data={"n_total": True, "n_correct": True, "seed_count": True, "temperature": True, "mean": ":.3f"},
    )
    fig.update_layout(
        # title=title or f"{metric} vs Temperature (seed-pooled; 95% CIs)",
        yaxis_title=metric,
        xaxis_title="Temperature",
        legend_title="Model",
        hovermode="x unified",
    )
    # Make all facet y-axes labeled 'Accuracy'
    for ax in fig.layout:
        if isinstance(fig.layout[ax], dict) and ax.startswith("yaxis"):
            fig.layout[ax]["title"] = {"text": "Accuracy"}
    return fig


import plotly.express as px

def plot_passk_vs_temperature_avg(agg_df, facet="dataset", title=None, use_ci=True):
    """
    Plot all detected pass@k metrics vs temperature in one figure (long form),
    with distinct colors per model_base and distinct line styles per pass@k.
    """
    pass_cols = [c for c in agg_df["metric"].unique() if c.startswith("pass@")]
    if not pass_cols:
        raise ValueError("No pass@k metrics found in agg_df.")

    data = agg_df[agg_df["metric"].isin(pass_cols)].copy()

    # --- legend-friendly series names (still useful for hover) ---
    data["series"] = data["model_base"] + " · " + data["metric"]

    # --- BIG non-repeating qualitative palette so colors don't duplicate ---
    big_palette = (
        px.colors.qualitative.Dark24
        + px.colors.qualitative.Light24
        + px.colors.qualitative.Alphabet
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Bold
        + px.colors.qualitative.Prism
        + px.colors.qualitative.Safe
        + px.colors.qualitative.Vivid
        + px.colors.qualitative.T10
        + px.colors.qualitative.G10
    )
    models = sorted(data["model_base"].unique())
    # If there are more models than colors, palette will repeat—but this gives you
    # >150 distinct colors before any repetition.
    color_map = {m: big_palette[i % len(big_palette)] for i, m in enumerate(models)}

    # --- Line styles per pass@k ---
    dash_styles_cycle = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    metrics = sorted(data["metric"].unique(), key=lambda s: (len(s), s))
    line_dash_map = {m: dash_styles_cycle[i % len(dash_styles_cycle)] for i, m in enumerate(metrics)}

    fig = px.line(
        data.sort_values([facet, "model_base", "metric", "temperature"]),
        x="temperature",
        y="mean",
        color="model_base",                     # color by model (no color dupes)
        line_dash="metric",                     # different line type per pass@k
        color_discrete_map=color_map,
        line_dash_map=line_dash_map,
        facet_col=facet,
        facet_col_wrap=2 if data[facet].nunique() > 2 else 0,
        markers=True,
        error_y="ci95" if use_ci else None,
        hover_data={
            "series": True,
            "model_base": True,
            "metric": True,
            "n_total": True,
            "n_correct": True,
            "seed_count": True,
            "temperature": True,
            "mean": ":.3f",
        },
        category_orders={
            "model_base": models,
            "metric": metrics,
        },
    )

    fig.update_layout(
        # title=title or "pass@k vs Temperature (seed-pooled; 95% CIs)",
        yaxis_title="Accuracy",
        xaxis_title="Temperature",
        legend_title="Color: Model · Line: pass@k",
        hovermode="x unified",
    )
    # Slightly thicker lines for readability
    fig.update_traces(mode="lines+markers", line=dict(width=2))

    return fig

def aggregate_for_table(summary_df):
    df = summary_df.copy()
    # expected columns: dataset, model, best_top1, best_sc, n_prompts, best_temp_top1, best_temp_sc, top1_CI95 (string)
    # Add model_base + seed
    df[["model_base","seed"]] = df["model"].apply(lambda m: pd.Series(split_model_seed(m)))

    # If best_top1 is a float already, great. If it's a string like "0.418 [0.39, 0.44]" split it:
    if df["best_top1"].dtype == object:
        # try to extract float at start of the string
        def _to_float(x):
            try:
                return float(str(x).split()[0])
            except Exception:
                return np.nan
        df["best_top1_val"] = df["best_top1"].apply(_to_float)
    else:
        df["best_top1_val"] = df["best_top1"].astype(float)

    # For pooled binom, we need counts. If you want to treat best_top1 at its *best_temp_top1* count,
    # reuse n_prompts as denominator (you used same prompts per run)
    df["n_for_top1"] = df["n_prompts"].astype(int)

    # Prepare aggregation rows
    rows = []
    for (dataset, model_base), g in df.groupby(["dataset","model_base"]):
        # Pool Top-1 across seeds
        top1, lo, hi, n_correct, n_total = _pooled_binom(g["best_top1_val"], g["n_for_top1"])

        # SC: report mean across seeds (you can pool as binom if you have SC counts; here we show mean ± std)
        sc_mean = g["best_sc"].mean() if "best_sc" in g.columns else np.nan
        sc_std  = g["best_sc"].std(ddof=1) if "best_sc" in g.columns else np.nan

        # Temperatures: mode (most frequent) of best temps
        def _mode(s):
            s = s.dropna()
            if s.empty: return np.nan
            return s.mode().iloc[0]
        t_top1 = _mode(g["best_temp_top1"]) if "best_temp_top1" in g.columns else np.nan
        t_sc   = _mode(g["best_temp_sc"])   if "best_temp_sc"   in g.columns else np.nan

        seed_count = g["seed"].nunique() if g["seed"].notna().any() else g["model"].nunique()

        rows.append(dict(
            dataset=dataset,
            model=model_base,
            seeds=seed_count,
            top1=top1,
            top1_lo=lo,
            top1_hi=hi,
            n_total=n_total,
            sc_mean=sc_mean,
            sc_std=sc_std,
            t_top1=t_top1,
            t_sc=t_sc
        ))
    agg = pd.DataFrame(rows)

    # Pretty columns for LaTeX
    def fmt_ci(row):
        if np.isnan(row["top1"]): return "--"
        # show as 0.418 [0.39, 0.44]
        return f"{row['top1']:.3f} [{row['top1_lo']:.2f}, {row['top1_hi']:.2f}]"

    def fmt_sc(row):
        if np.isnan(row["sc_mean"]): return "--"
        if np.isnan(row["sc_std"]) or row["seeds"] <= 1:
            return f"{row['sc_mean']:.3f}"
        return f"{row['sc_mean']:.3f} ± {row['sc_std']:.3f}"

    agg["Top-1 (95% CI)"] = agg.apply(fmt_ci, axis=1)
    agg["SC@best"] = agg.apply(fmt_sc, axis=1)
    agg["T* (Top-1)"] = agg["t_top1"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "--")
    agg["T* (SC)"]    = agg["t_sc"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "--")

    # Order methods nicely
    order = ["baseline", "clip", "default", "smoothing_ratio"]
    agg["model_order"] = agg["model"].apply(lambda m: (order.index(m) if m in order else 99, m))
    agg = agg.sort_values(["dataset","model_order","model"]).drop(columns=["model_order"])

    # One table per dataset (nice for the paper)
    tables = {ds: sub[["model","seeds","Top-1 (95% CI)","SC@best","T* (Top-1)","T* (SC)"]]
              for ds, sub in agg.groupby("dataset")}
    return tables, agg

# ---------------------------
# Example usage
# ---------------------------

RESULTS_DIR_TOKEN = "eval_16_results"

if __name__ == "__main__":
    ROOT = "/scratch/mcd2g19/grpo_test/cached_files/gsm8k_param_models" 
    # ROOT = "/scratch/mcd2g19/grpo_test/cached_files/gsm8k_Qwen2.5-1.5B_ft_models" 

    # try:
    #     samples_df = pd.read_csv(f'{ROOT}/plots/samples_df.csv')
    #     attempts_df = pd.read_csv(f'{ROOT}/plots/attempts_df.csv')
    # except Exception:
    samples_df, attempts_df = load_runs(ROOT)

    samples_df.to_csv(f'{ROOT}/plots/samples_df.csv')
    attempts_df.to_csv(f'{ROOT}/plots/attempts_df.csv')

    # run_reasoning_analyses(attempts_df)
    # quit()

    # copy_attempts = attempts_df.copy()
    # model_change_dict = {}
    # # randomly rename models, keeping within seed name consistent
    # model_names = ["GRPO-clip", "baseline", "GRPO-noclip", "GR-PSPO"]
    # random.shuffle(model_names)

    # new_model_names = ["A", "B", "C", "D"]

    # model_change_dict = {}
    # for i, model_name in enumerate(model_names):
    #     for seed in [0, 1, 2, 3, 4]:
    #         old_model_name = f"{model_name}_{seed}"
    #         new_model_name = f"{new_model_names[i]}_{seed}"

    #         model_change_dict[old_model_name] = new_model_name

    # print(model_change_dict)
    
    copy_attempts = df[df["model"].endswith("_0")]
    # print(copy_attempts.columns.to_list())
    for dataset in ["gsm8k", "MATH-500", "SVAMP", "asdiv"]:
        randomly_select_prompts(copy_attempts, n=20, dataset=dataset, temperature=0.0, blind_models=True)

    # for dataset in ["gsm8k", "MATH-500", "SVAMP", "asdiv"]:
    #     polarising_df = find_polarising_prompts(
    #         attempts_df, n=5, dataset=dataset, temperature=0.0
    #     )

    #     print_polarising_responses(polarising_df)
    seen = samples_df.groupby(["dataset","model"])["prompt"].nunique().reset_index(name="n_prompts_seen")
    print(seen.sort_values(["dataset","model"]).to_string(index=False))

    # Choose your grouping granularity
    group_cols = ["dataset", "model", "temperature"]
    summary_df = summarize_by(group_cols, samples_df, attempts_df, passk_list=(1,2,4,8,16))

    if MAKE_PLOTS:
        # Show / save figures
        figs = {
            "accuracy_vs_temp": plot_accuracy_vs_temperature(summary_df, facet="dataset"),
            "passk_vs_temp": plot_passk_vs_temperature(summary_df, facet="dataset"),
            # "reward_violin": plot_reward_distributions(attempts_df, facet="dataset"),
            # "diversity_vs_temp": plot_diversity_vs_temp(summary_df, facet="dataset"),
            # "len_vs_reward": plot_len_vs_reward(attempts_df, facet="dataset"),
        }

        os.makedirs(f"{ROOT}/figs", exist_ok=True)
        for name, fig in figs.items():
            image_path = os.path.join(f"{ROOT}/figs", name)
            safe_write_image(fig, image_path)

# Use your existing summary_df (one row per dataset/model/temperature/seed)
# It should include: dataset, model (e.g., 'smoothing_ratio_0'), temperature, n_prompts (or n_scorable),
# and metrics like: top1_acc, bestofn_acc, majority_vote_acc, pass@1, pass@2, pass@4 (if available)

    agg = aggregate_over_seeds(summary_df)

    for temperature in [0.0, 0.2, 0.4, 0.6, 0.8]:
        print(f"Temperature: {temperature}")
        new_df = agg[agg["temperature"]==temperature]
        new_df = new_df.drop(columns=["temperature", "seed_count"])
        print(new_df.to_latex(index=False, float_format="%.3f"))
        print()

    # print(agg.to_latex(index=False, float_format="%.3f"))
    # print(agg.to_string(index=False))

    # tables = build_latex_tables(
    #     aggregate_over_seeds,
    #     out_dir="{ROOT}/tables",
    #     as_percent=False  # set True for percentage formatting
    # )
    # `tables` is a dict: {dataset -> LaTeX string}
    # print(next(iter(tables.values())))

    if MAKE_PLOTS:
        # diff_cat_all = diff_heatmaps_per_dataset(agg, axis="category", baseline_model=None, facet_wrap=2, symmetric_range=0.25)

        # # Save
        # os.makedirs(f"{ROOT}/figs/category_heatmaps", exist_ok=True)
        # for ds, fig in diff_cat_all.items():
        #     if ds.lower() != "gsm8k":
        #         image_path = os.path.join(f"{ROOT}/figs/category_heatmaps/{ds}_category_diff")
        #         safe_write_image(fig, image_path)

        # diff_dif_all = diff_heatmaps_per_dataset(agg, axis="difficulty", baseline_model=None, facet_wrap=2, symmetric_range=0.25)

        # os.makedirs(f"{ROOT}/figs/difficulty_heatmaps", exist_ok=True)
        # for ds, fig in diff_dif_all.items():
        #     if ds.lower() == "math-500":
        #         image_path = os.path.join(f"{ROOT}/figs/difficulty_heatmaps/{ds}_difficulty_diff")
        #         safe_write_image(fig, image_path)

        # figs = plot_category_heatmap_per_dataset(agg, facet_wrap=3, set_max=False)
        # os.makedirs(f"{ROOT}/figs/category_heatmaps", exist_ok=True)
        # for ds, fig in figs.items():
        #     if ds.lower() != "gsm8k":
        #         image_path = os.path.join(f"{ROOT}/figs/category_heatmaps/{ds}_category_heatmap")
        #         safe_write_image(fig, image_path)

        # diff_figs = plot_difficulty_heatmap_per_dataset(agg, facet_wrap=3, set_max=False)
        # os.makedirs(f"{ROOT}/figs/difficulty_heatmaps", exist_ok=True)
        # for ds, fig in diff_figs.items():
        #     if ds.lower() == "math-500":
        #         image_path = os.path.join(f"{ROOT}/figs/difficulty_heatmaps/{ds}_difficulty_heatmap")
        #         safe_write_image(fig, image_path)
        
        print(set(agg["metric"]))
        # 1) Top-1 accuracy vs temperature
        for metric in set(agg["metric"]):
            fig = plot_metric_vs_temperature_avg(agg, metric=metric, facet="dataset", title="Top-1 vs Temperature")
            image_path = os.path.join(f"{ROOT}/figs/avg_{metric}_vs_temp")
            safe_write_image(fig, image_path)

        # fig = plot_metric_vs_temperature_avg(agg, metric="gen_len_mean", facet="dataset", title="Top-1 vs Temperature")
        # image_path = os.path.join(f"{ROOT}/figs/avg_acc_vs_temp")
        # safe_write_image(fig, image_path)

        # # 2) Self-consistency (if you use 'bestofn_acc' as SC@N)
        # if "bestofn_acc" in agg["metric"].unique():
        #     fig = plot_metric_vs_temperature_avg(agg, metric="bestofn_acc", facet="dataset", title="SC@N vs Temperature")
        #     image_path = os.path.join(f"{ROOT}/figs/avg_bestofn_acc_vs_temp")
        #     safe_write_image(fig, image_path)

        # 3) pass@k (plots all pass@k columns found)
        try:
            fig = plot_passk_vs_temperature_avg(agg, facet="dataset", title="pass@k vs Temperature")
            image_path = os.path.join(f"{ROOT}/figs/avg_passk_vs_temp")
            safe_write_image(fig, image_path)
        except ValueError:
            pass  # no pass@k columns present

    # print(agg)

    # If you want a single combined table (multi-dataset), you can concat with a dataset column:
    # combined = agg[["dataset","model","seeds","Top-1 (95% CI)","SC@best","T* (Top-1)","T* (SC)"]].copy()
    # print(combined.to_latex(index=False, escape=False, longtable=True,
    #                         caption="All datasets: Qwen2.5-0.5B results aggregated across seeds.",
    #                         label="tab:all-qwen05",
    #                         column_format="l l r l l c c"))

    # Also handy: a tidy table for quick inspection
    cols = ["dataset","model","temperature","n_prompts",
            "top1_acc","bestofn_acc","majority_vote_acc","reward_pick_acc",
            "pass@1","pass@2","pass@4",
            "reward_mean","reward_std","gen_len_mean","uniq_answers_mean","agreement_mean",
            "reward_correct_corr"]
    # print(summary_df[cols].sort_values(["dataset","model","temperature"]))

    table = summary_df[cols].sort_values(["dataset","model","temperature"]).round(3)
    print(table.to_string(index=False, max_rows=None, max_cols=None))

    # print(f"Summary table with comparison model as the baseline")
    # summary = publication_ready_summary(summary_df, gsm8k_model="baseline_0")
    # print(summary.to_string(index=False))

    # print(f"Summary tables with comparison model as the baseline")
    # for temp in [0.0, 0.2, 0.4, 0.6, 0.8]:
    #     temp_df = summary_df[summary_df["temperature"] == 0.0]
    #     summary = publication_ready_summary(temp_df, gsm8k_model="baseline_0")
    #     print(summary.to_string(index=False))

    #     tables, agg = aggregate_for_table(temp_df)

    #     # Print LaTeX for each dataset
    #     for ds, tdf in tables.items():
    #         print(to_latex_table(tdf, ds))
    #         print()  # spacer


    # print(f"Summary table with comparison model as the default_0")
    # summary = publication_ready_summary(summary_df, gsm8k_model="default_0")
    # print(summary.to_string(index=False))
    # # 1) Create judge sheets
    # export_likert_and_arena(
    #     samples_df,
    #     out_dir=f"{ROOT}/qual_sheets",
    #     likert_kwargs=dict(
    #         datasets=None,            # e.g. ["mmlu"]
    #         temperatures=[0,0.2,0.4,0.8],
    #         n_per_category=10,
    #         hide_gold=True,           # hide gold for blind rating
    #     ),
    #     arena_kwargs=dict(
    #         datasets=None,            # e.g. ["mmlu"]
    #         temperatures=[0,0.2,0.4,0.8],
    #         pairs_per_prompt=1,
    #         n_prompts=20,
    #         hide_gold=True
    #     ),
    #     seed=42
    # )

    # 2) After annotating `qual_sheets/arena_sheet.csv` (fill winner A/B/Tie),
    #    analyze preferences:
    # winrate, scores = analyze_arena(f"{ROOT}/qual_sheets/arena_sheet_judged.csv")
    # print("\nPairwise win rates (row vs column):")
    # print(winrate.round(3).to_string())
    # print("\nBradley–Terry scores:")
    # print(scores.round(3).to_string(index=False))
