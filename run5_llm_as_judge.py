
#!/usr/bin/env python3
"""
LLM-as-judge evaluation of prototype interpretability on the corrected prompts.

For each test instance the LLM compares two prototype methods (Stage A vs Stage B),
returning a JSON object with concept counts, overlap with the test instance,
irrelevant-feature counts, a most-similar verdict, and a confidence.

Runs across 5 models, summarizes each model's Stage A vs Stage B comparison,
aggregates mean / s.e.m. across models, and saves a bar chart as JPEG in plots/.

By default it evaluates a STRATIFIED 50% random sample of the data, drawn evenly
within each (dataset, model, seed) group, to keep the full run near ~10 hours.

Usage:
    python run6_llm_as_judge.py                 # 50% stratified sample (default)
    python run6_llm_as_judge.py --full          # use ALL rows
    python run6_llm_as_judge.py --frac 0.25     # custom sampling fraction
    python run6_llm_as_judge.py --sanity        # quick 30-instance sanity check
"""

import os
import re
import json
import time
import argparse
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # headless backend, safe for servers / no display
import matplotlib.pyplot as plt

from src.llm_query import call_llm


# Models to evaluate
MODELS = ['claude4.5-sonnet', 'gpt5', 'gpt5-mini', 'gpt-4o', 'o4-mini']

# Number of instances to use in sanity-check mode
SANITY_N = 30

# Default fraction of data to sample (stratified by dataset/model/seed)
DEFAULT_SAMPLE_FRAC = 0.5

# Reproducible sampling
RANDOM_STATE = 42

# Stratification columns
STRATA_COLS = ['dataset', 'model', 'seed']

# Output directory for plots
PLOTS_DIR = 'plots'

# Prompt column in the corrected CSV
PROMPT_COL = 'prompt'  # adjust if your column has a different name

# Expected integer keys in the JSON payload
INT_KEYS = [
    'stage_a_concepts_count',
    'stage_b_concepts_count',
    'stage_a_concepts_in_test',
    'stage_b_concepts_in_test',
    'stage_a_irrelevant_features',
    'stage_b_irrelevant_features',
]


def stratified_sample(df: pd.DataFrame, frac: float,
                      strata_cols: List[str], random_state: int) -> pd.DataFrame:
    """
    Randomly sample `frac` of the rows WITHIN each (dataset, model, seed) group,
    so every combination stays proportionally represented.

    Uses the built-in groupby(...).sample(), which preserves the grouping
    columns (unlike groupby.apply on pandas >= 2.2).

    Falls back to a global sample if none of the strata columns are present.
    """
    present = [c for c in strata_cols if c in df.columns]
    missing = [c for c in strata_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: strata columns not found and ignored: {missing}")

    if not present:
        print("  WARNING: no strata columns present -> using a global random sample.")
        return df.sample(frac=frac, random_state=random_state).reset_index(drop=True)

    print(f"  Stratifying {frac:.0%} sample by: {present}")

    # Built-in groupby sample: keeps all columns, samples within each group.
    sampled = (
        df.groupby(present, dropna=False, group_keys=False)
          .sample(frac=frac, random_state=random_state)
          .reset_index(drop=True)
    )

    # Report per-stratum counts so you can confirm coverage.
    print("\n  Sample coverage (rows kept / total) per stratum:")
    orig_counts = df.groupby(present, dropna=False).size()
    samp_counts = sampled.groupby(present, dropna=False).size()
    for key, total in orig_counts.items():
        kept = int(samp_counts.get(key, 0))
        key_tuple = key if isinstance(key, tuple) else (key,)
        label = " / ".join(str(k) for k in key_tuple)
        print(f"    {label:40s} {kept:>5d} / {total:<5d}")

    return sampled

def extract_json(response: str) -> Optional[dict]:
    # Robustly extract a JSON object from an LLM response.
    # Handles fenced code blocks, raw braces, and minor trailing text.
    # Returns a dict, or None if parsing fails.
    if not isinstance(response, str) or response.startswith('ERROR:'):
        print(f"Warning: invalid or error response: {response}")
        return None

    candidates = []

    # Prefer a fenced code block. The fence char (backtick) is built from
    # chr(96) so no literal backtick ever appears in this file.
    fence_tok = chr(96) * 3
    fence_pattern = fence_tok + r"(?:json)?\s*(\{.*?\})\s*" + fence_tok
    fence = re.search(fence_pattern, response, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))

    # Fall back to the last brace-delimited block in the text.
    brace_matches = re.findall(r"\{.*?\}", response, re.DOTALL)
    candidates.extend(reversed(brace_matches))

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    print("Warning: could not extract JSON from response")
    return None


def parse_record(response: str) -> Dict:
    """Parse one LLM response into a normalized record with a parse_ok flag."""
    obj = extract_json(response)
    record = {k: np.nan for k in INT_KEYS}
    record['most_similar_prototype'] = None
    record['confidence'] = np.nan
    record['parse_ok'] = False

    if obj is None:
        return record

    record['parse_ok'] = True
    for k in INT_KEYS:
        try:
            record[k] = int(obj.get(k)) if obj.get(k) is not None else np.nan
        except (ValueError, TypeError):
            record[k] = np.nan

    verdict = obj.get('most_similar_prototype')
    if isinstance(verdict, str):
        v = verdict.strip().lower()
        if v in ('stage_a', 'stage_b'):
            record['most_similar_prototype'] = v

    try:
        conf = float(obj.get('confidence'))
        if 0.0 <= conf <= 1.0:
            record['confidence'] = conf
    except (ValueError, TypeError):
        pass

    return record


def evaluate_model(df: pd.DataFrame, model_name: str, batch_size: int = 10) -> Dict:
    """Run the batched evaluation for a single model."""
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size

    print("\n" + "#" * 80)
    print(f"# MODEL: {model_name}")
    print("#" * 80)
    print(f"Processing {total_rows} rows in {num_batches} batches of {batch_size}")
    print(f"This will make {total_rows} total LLM queries\n")
    print("-" * 80)

    records: List[Dict] = []
    start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_rows)
        df_batch = df.iloc[batch_start:batch_end]

        batch_num = batch_idx + 1
        print(f"\n[{model_name}] Batch {batch_num}/{num_batches} (rows {batch_start+1}-{batch_end})")
        print("  Evaluating prototype-comparison prompts...")

        prompts = df_batch[PROMPT_COL].tolist()
        responses = call_llm(prompts, model_name=model_name, temperature=1.0)
        records.extend([parse_record(r) for r in responses])

        parsed = sum(1 for r in records if r['parse_ok'])
        a_votes = sum(1 for r in records if r['most_similar_prototype'] == 'stage_a')
        b_votes = sum(1 for r in records if r['most_similar_prototype'] == 'stage_b')
        print(f"  Running: parsed={parsed}/{len(records)} | "
              f"Stage A picked={a_votes}, Stage B picked={b_votes}")

        elapsed = time.time() - start_time
        eta_min = (num_batches - batch_num) * (elapsed / batch_num) / 60
        print(f"  Time elapsed: {elapsed/60:.1f} min | ETA: {eta_min:.1f} min")
        print("-" * 80)

    rec_df = pd.DataFrame(records)
    results_df = df.reset_index(drop=True).copy()
    for col in rec_df.columns:
        results_df[col] = rec_df[col].values

    summary = summarize_model(results_df, model_name)

    safe_name = model_name.replace('.', '_').replace('/', '_')
    suffix = '_sanity' if total_rows <= SANITY_N else ''
    out_path = f'prototype_evaluation_results_{safe_name}{suffix}.csv'
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")

    return {'model': model_name, 'results_df': results_df, 'summary': summary}


def summarize_model(results_df: pd.DataFrame, model_name: str) -> Dict:
    """Compute and print a Stage A vs Stage B summary for one model."""
    n = len(results_df)
    parsed = int(results_df['parse_ok'].sum())
    valid = results_df[results_df['most_similar_prototype'].notna()]

    a_votes = int((valid['most_similar_prototype'] == 'stage_a').sum())
    b_votes = int((valid['most_similar_prototype'] == 'stage_b').sum())
    n_valid = len(valid)
    a_share = a_votes / n_valid if n_valid else 0.0
    b_share = b_votes / n_valid if n_valid else 0.0

    def m(col):
        return float(results_df[col].mean(skipna=True))

    summary = {
        'model': model_name,
        'n': n,
        'parsed': parsed,
        'parse_rate': parsed / n if n else 0.0,
        'n_valid_verdicts': n_valid,
        'stage_a_picked': a_votes,
        'stage_b_picked': b_votes,
        'stage_a_share': a_share,
        'stage_b_share': b_share,
        'mean_a_concepts': m('stage_a_concepts_count'),
        'mean_b_concepts': m('stage_b_concepts_count'),
        'mean_a_in_test': m('stage_a_concepts_in_test'),
        'mean_b_in_test': m('stage_b_concepts_in_test'),
        'mean_a_irrelevant': m('stage_a_irrelevant_features'),
        'mean_b_irrelevant': m('stage_b_irrelevant_features'),
        'mean_confidence': m('confidence'),
    }
    summary['mean_a_coverage'] = (
        summary['mean_a_in_test'] / summary['mean_a_concepts']
        if summary['mean_a_concepts'] else 0.0
    )
    summary['mean_b_coverage'] = (
        summary['mean_b_in_test'] / summary['mean_b_concepts']
        if summary['mean_b_concepts'] else 0.0
    )

    print("\n" + "=" * 80)
    print(f"SUMMARY — {model_name}  (Stage A vs Stage B)")
    print("=" * 80)
    print(f"  Parsed responses:       {parsed}/{n} ({summary['parse_rate']:.1%})")
    print(f"  Valid verdicts:         {n_valid}")
    print(f"  Most-similar verdict:   Stage A {a_votes} ({a_share:.1%}) | "
          f"Stage B {b_votes} ({b_share:.1%})")
    print(f"  Mean confidence:        {summary['mean_confidence']:.2f}")
    print(f"\n  Concept analysis (mean per instance):")
    print(f"    {'Metric':28s} {'Stage A':>10s} {'Stage B':>10s}")
    print(f"    {'concepts identified':28s} {summary['mean_a_concepts']:>10.2f} "
          f"{summary['mean_b_concepts']:>10.2f}")
    print(f"    {'concepts present in test':28s} {summary['mean_a_in_test']:>10.2f} "
          f"{summary['mean_b_in_test']:>10.2f}")
    print(f"    {'irrelevant features':28s} {summary['mean_a_irrelevant']:>10.2f} "
          f"{summary['mean_b_irrelevant']:>10.2f}")
    print(f"    {'concept coverage':28s} {summary['mean_a_coverage']:>10.1%} "
          f"{summary['mean_b_coverage']:>10.1%}")
    return summary


def plot_results(summaries: List[Dict], mean_share_a: float, sem_share_a: float,
                 sanity: bool = False) -> str:
    """Grouped bar chart of Stage A vs Stage B verdict share per model + Mean."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    labels = [s['model'] for s in summaries] + ['Mean']
    a_shares = [s['stage_a_share'] for s in summaries] + [mean_share_a]
    b_shares = [s['stage_b_share'] for s in summaries] + [1.0 - mean_share_a]
    a_err = [0.0] * len(summaries) + [sem_share_a]
    b_err = [0.0] * len(summaries) + [sem_share_a]

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    ba = ax.bar(x - w / 2, a_shares, w, yerr=a_err, capsize=6,
                label='Stage A picked', color='#4C72B0', edgecolor='black', linewidth=0.7)
    bb = ax.bar(x + w / 2, b_shares, w, yerr=b_err, capsize=6,
                label='Stage B picked', color='#C44E52', edgecolor='black', linewidth=0.7)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    for bars in (ba, bb):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{bar.get_height():.0%}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Share of "most similar" verdicts')
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    title = 'Prototype Most-Similar Verdicts by Model (Stage A vs Stage B)'
    if sanity:
        title += f' (SANITY, n={SANITY_N})'
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.text(0.99, 0.02,
            f'Stage A share: {mean_share_a:.1%} ± {sem_share_a:.1%} (mean ± s.e.m., n={len(summaries)})',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    fig.tight_layout()

    suffix = '_sanity' if sanity else ''
    out_path = os.path.join(PLOTS_DIR, f'prototype_evaluation_verdicts{suffix}.jpeg')
    fig.savefig(out_path, format='jpeg', dpi=200)
    plt.close(fig)
    print(f"\n  Plot saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge prototype evaluation (corrected prompts).")
    parser.add_argument('--sanity', action='store_true',
                        help=f'Quick sanity check on the first {SANITY_N} instances.')
    parser.add_argument('--full', action='store_true',
                        help='Use ALL rows instead of the default stratified sample.')
    parser.add_argument('--frac', type=float, default=DEFAULT_SAMPLE_FRAC,
                        help=f'Stratified sampling fraction (default {DEFAULT_SAMPLE_FRAC}).')
    args = parser.parse_args()

    csv_path = 'prototype_evaluation_prompts_corrected.csv'
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    if PROMPT_COL not in df.columns:
        print(f"ERROR: '{PROMPT_COL}' column not found in CSV")
        print("Available columns:", df.columns.tolist())
        return

    df = df.dropna(subset=[PROMPT_COL])
    df = df[df[PROMPT_COL].astype(str).str.strip() != ''].reset_index(drop=True)
    print(f"After filtering for valid prompts: {len(df)} rows")
    if len(df) == 0:
        print("ERROR: No valid prompts found!")
        return

    # ---- Stratified sampling (default), unless --full or --sanity ----
    if args.sanity:
        df = df.head(SANITY_N).reset_index(drop=True)
        print("\n" + "*" * 80)
        print(f"* SANITY-CHECK MODE: {len(df)} instances x {len(MODELS)} models "
              f"= {len(df) * len(MODELS)} queries")
        print("*" * 80)
    elif args.full:
        print("\nUsing the FULL dataset (no sampling).")
    else:
        print(f"\nDrawing a stratified {args.frac:.0%} sample (seed={RANDOM_STATE}):")
        before = len(df)
        df = stratified_sample(df, args.frac, STRATA_COLS, RANDOM_STATE)
        print(f"\n  Sampled {len(df)} of {before} rows "
              f"({len(df)/before:.1%}) across {len(MODELS)} models "
              f"= {len(df) * len(MODELS)} total queries")

    total_rows = len(df)
    if total_rows == 0:
        print("ERROR: No rows left after sampling!")
        return

    # ---- Run every model ----
    per_model = {m: evaluate_model(df, m) for m in MODELS}
    summaries = [per_model[m]['summary'] for m in MODELS]

    # ---- Side-by-side model comparison table ----
    print("\n" + "=" * 80)
    print("MODEL COMPARISON — Stage A vs Stage B")
    print("=" * 80)
    header = (f"{'Model':18s} {'A picked':>9s} {'B picked':>9s} "
              f"{'A cover':>8s} {'B cover':>8s} {'A irrel':>8s} {'B irrel':>8s} {'conf':>6s}")
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(f"{s['model']:18s} {s['stage_a_share']:>8.1%} {s['stage_b_share']:>8.1%} "
              f"{s['mean_a_coverage']:>7.1%} {s['mean_b_coverage']:>7.1%} "
              f"{s['mean_a_irrelevant']:>8.2f} {s['mean_b_irrelevant']:>8.2f} "
              f"{s['mean_confidence']:>6.2f}")

    # ---- Aggregate across models: mean / s.e.m. of Stage A share ----
    a_shares = np.array([s['stage_a_share'] for s in summaries], dtype=float)
    n_models = len(a_shares)
    mean_share_a = a_shares.mean()
    std_share_a = a_shares.std(ddof=1) if n_models > 1 else 0.0
    sem_share_a = std_share_a / np.sqrt(n_models) if n_models > 1 else 0.0

    print("\n" + "=" * 80)
    print("AGGREGATED ACROSS MODELS (mean / s.e.m.)")
    print("=" * 80)
    print(f"  Stage A 'most similar' share: {mean_share_a:.2%} ± {sem_share_a:.2%} "
          f"(mean ± s.e.m., n={n_models})")
    print(f"  Stage B 'most similar' share: {1-mean_share_a:.2%} ± {sem_share_a:.2%}")

    # ---- Save aggregate summary ----
    summary_suffix = '_sanity' if args.sanity else ''
    pd.DataFrame(summaries).to_csv(
        f'prototype_evaluation_summary{summary_suffix}.csv', index=False)
    print(f"\n  Per-model summary saved to prototype_evaluation_summary{summary_suffix}.csv")

    # ---- Plot ----
    plot_results(summaries, mean_share_a, sem_share_a, sanity=args.sanity)

    # ---- Interpretation ----
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(f"\nAcross {n_models} models, the compact Stage A prototype was judged most")
    print(f"similar in {mean_share_a:.1%} ± {sem_share_a:.1%} of cases (mean ± s.e.m.).")
    print("Compare each method's concept-coverage and irrelevant-feature counts above:")
    print("if Stage A's coverage is comparable to Stage B's while carrying fewer")
    print("irrelevant features, that supports the claim that the reduced prototype")
    print("size does NOT make similarity harder to assess.")
    if args.sanity:
        print("\n[SANITY CHECK COMPLETE] If this looks right, rerun without --sanity.")
    print("\n" + "=" * 80)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
