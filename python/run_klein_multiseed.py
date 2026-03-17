#!/usr/bin/env python3
"""
Multi-seed Klein bottle experiment runner.
Runs klein_experiment.py with 3 seeds, then aggregates results.

Usage:
    # Full run (matches paper settings)
    python run_klein_multiseed.py --full --device cuda

    # Quick sanity check
    python run_klein_multiseed.py --quick --device cuda

    # Aggregate only (if seeds already ran)
    python run_klein_multiseed.py --aggregate-only
"""

import json
import os
import sys
import subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict

SEEDS = [42, 179, 316]
SCRIPT = os.path.join(os.path.dirname(__file__), 'klein_experiment.py')
RESULTS_DIR = 'results_klein_multiseed'
TEST_LENGTHS = [2, 3, 4, 6, 8, 10]

TYPE_LABELS = {
    'klein_cover': 'A',
    'klein_transport': 'B',
    'klein_homotopy': 'B',
    'klein_transformer': 'A',
    'klein_transformer_wc': 'A',
}


def run_single_seed(seed, args):
    """Run one seed of the Klein bottle experiment."""
    output_dir = os.path.join(RESULTS_DIR, f'seed_{seed}')
    figure_dir = os.path.join(RESULTS_DIR, f'figures_seed_{seed}')

    cmd = [
        sys.executable, SCRIPT,
        '--seed', str(seed),
        '--output_dir', output_dir,
        '--figure_dir', figure_dir,
        '--device', args.device,
        '--epochs', str(args.epochs),
        '--n_samples', str(args.n_samples),
        '--n_avg', str(args.n_avg),
        '--n_points', str(args.n_points),
    ]
    if args.archs:
        cmd += ['--archs'] + args.archs

    print(f"\n{'='*70}")
    print(f"  KLEIN BOTTLE — SEED {seed}  ({SEEDS.index(seed)+1}/{len(SEEDS)})")
    print(f"{'='*70}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: seed {seed} failed with return code {result.returncode}")
        return None

    json_path = os.path.join(output_dir, f'klein_seed{seed}.json')
    if not os.path.exists(json_path):
        print(f"ERROR: expected output not found: {json_path}")
        return None

    with open(json_path) as f:
        return json.load(f)


def aggregate_results(all_results):
    """Aggregate per-seed results into mean±std tables."""
    archs = list(all_results[0].keys())
    n_seeds = len(all_results)

    print(f"\n{'='*80}")
    print(f"AGGREGATED KLEIN BOTTLE RESULTS (mean ± std over {n_seeds} seeds)")
    print(f"{'='*80}")

    # ---- Training Loss ----
    print(f"\n--- Training Loss ---")
    print(f"{'Architecture':<25} {'Mean':>10} {'Std':>10} {'Seeds':>30}")
    print("-" * 80)
    for arch in archs:
        losses = []
        for res in all_results:
            if arch in res:
                losses.append(res[arch].get('best_loss', res[arch].get('final_loss', float('nan'))))
        if losses:
            seeds_str = str([f"{v:.4f}" for v in losses])
            print(f"{arch:<25} {np.mean(losses):10.4f} {np.std(losses):10.4f} {seeds_str}")

    # ---- Per-Segment Chamfer (ALL words) ----
    print(f"\n--- Per-Segment Chamfer Distance — ALL words (mean ± std) ---")
    header = f"{'Architecture':<25} {'Type':>5}"
    for L in TEST_LENGTHS:
        header += f"{'L='+str(L):>12}"
    print(header)
    print("-" * len(header))

    agg = {}
    for arch in archs:
        typ = TYPE_LABELS.get(arch, '?')
        row = f"{arch:<25} {typ:>5}"
        agg[arch] = {'type': typ}

        for L in TEST_LENGTHS:
            vals = []
            for res in all_results:
                if arch in res:
                    t3 = res[arch].get('battery', {}).get('t3_scaling', {})
                    v = t3.get(str(L), t3.get(L, {}))
                    if isinstance(v, dict):
                        ps = v.get('per_seg_chamfer', float('nan'))
                        if not np.isnan(ps):
                            vals.append(ps)
            if vals:
                m, s = np.mean(vals), np.std(vals)
                row += f" {m:.3f}±{s:.3f}"
                agg[arch][L] = {'mean': m, 'std': s, 'vals': vals}
            else:
                row += f"{'—':>12}"
        print(row)

    # ---- Per-Segment Chamfer (NON-CANONICAL only) ----
    print(f"\n--- Per-Segment Chamfer — NON-CANONICAL words only (exercises relation) ---")
    print(header)
    print("-" * len(header))

    for arch in archs:
        typ = TYPE_LABELS.get(arch, '?')
        row = f"{arch:<25} {typ:>5}"
        for L in TEST_LENGTHS:
            vals = []
            for res in all_results:
                if arch in res:
                    t3 = res[arch].get('battery', {}).get('t3_scaling', {})
                    v = t3.get(str(L), t3.get(L, {}))
                    if isinstance(v, dict):
                        ps = v.get('per_seg_noncanonical', float('nan'))
                        if not np.isnan(ps):
                            vals.append(ps)
            if vals:
                m, s = np.mean(vals), np.std(vals)
                row += f" {m:.3f}±{s:.3f}"
            else:
                row += f"{'—':>12}"
        print(row)

    # ---- Per-Segment Chamfer (CANONICAL only) ----
    print(f"\n--- Per-Segment Chamfer — CANONICAL words only ---")
    print(header)
    print("-" * len(header))

    for arch in archs:
        typ = TYPE_LABELS.get(arch, '?')
        row = f"{arch:<25} {typ:>5}"
        for L in TEST_LENGTHS:
            vals = []
            for res in all_results:
                if arch in res:
                    t3 = res[arch].get('battery', {}).get('t3_scaling', {})
                    v = t3.get(str(L), t3.get(L, {}))
                    if isinstance(v, dict):
                        ps = v.get('per_seg_canonical', float('nan'))
                        if not np.isnan(ps):
                            vals.append(ps)
            if vals:
                m, s = np.mean(vals), np.std(vals)
                row += f" {m:.3f}±{s:.3f}"
            else:
                row += f"{'—':>12}"
        print(row)

    # ---- Relation Gap ----
    print(f"\n--- Relation Gap (ba vs a⁻¹b) ---")
    print(f"{'Architecture':<25} {'Mean Fréchet':>15} {'Std':>10}")
    print("-" * 55)
    for arch in archs:
        vals = []
        for res in all_results:
            if arch in res:
                t1 = res[arch].get('battery', {}).get('t1_relation', {})
                mf = t1.get('mean_frechet', float('nan'))
                if not np.isnan(mf):
                    vals.append(mf)
        if vals:
            print(f"{arch:<25} {np.mean(vals):15.4f} {np.std(vals):10.4f}")

    # ---- Gap Summary ----
    print(f"\n--- Gap Summary at L=10 ---")
    type_b_vals = []
    type_a_vals = []
    for arch in archs:
        typ = TYPE_LABELS.get(arch, '?')
        vals = []
        for res in all_results:
            if arch in res:
                t3 = res[arch].get('battery', {}).get('t3_scaling', {})
                v = t3.get('10', t3.get(10, {}))
                if isinstance(v, dict):
                    ps = v.get('per_seg_chamfer', float('nan'))
                    if not np.isnan(ps):
                        vals.append(ps)
        if vals:
            mean_v = np.mean(vals)
            if typ == 'B':
                type_b_vals.append(mean_v)
            else:
                type_a_vals.append(mean_v)

    if type_b_vals:
        print(f"  Type-B range: {min(type_b_vals):.3f}–{max(type_b_vals):.3f}")
    if type_a_vals:
        print(f"  Type-A range: {min(type_a_vals):.3f}–{max(type_a_vals):.3f}")
    if type_b_vals and type_a_vals:
        gap = np.mean(type_a_vals) / np.mean(type_b_vals)
        print(f"  Gap ratio: {gap:.1f}×")

    # ---- Homotopy vs Transport gap (non-canonical) ----
    print(f"\n--- Homotopy vs Transport Gap (non-canonical words at L=10) ---")
    transport_noncan = []
    homotopy_noncan = []
    for res in all_results:
        for arch in ['klein_transport', 'klein_homotopy']:
            if arch in res:
                t3 = res[arch].get('battery', {}).get('t3_scaling', {})
                v = t3.get('10', t3.get(10, {}))
                if isinstance(v, dict):
                    ps = v.get('per_seg_noncanonical', float('nan'))
                    if not np.isnan(ps):
                        if arch == 'klein_transport':
                            transport_noncan.append(ps)
                        else:
                            homotopy_noncan.append(ps)
    if transport_noncan and homotopy_noncan:
        t_mean = np.mean(transport_noncan)
        h_mean = np.mean(homotopy_noncan)
        print(f"  Transport (no H): {t_mean:.3f} ± {np.std(transport_noncan):.3f}")
        print(f"  Homotopy  (+ H):  {h_mean:.3f} ± {np.std(homotopy_noncan):.3f}")
        print(f"  H closes gap by:  {(t_mean - h_mean) / t_mean * 100:.1f}%")
        print(f"  THIS is the proof term made measurable.")

    # Save aggregated results
    agg_save = {'n_seeds': n_seeds, 'seeds': SEEDS, 'architectures': {}}
    for arch in archs:
        agg_save['architectures'][arch] = agg.get(arch, {})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'aggregated.json'), 'w') as f:
        json.dump(agg_save, f, indent=2, default=str)
    print(f"\nAggregated results saved to: {RESULTS_DIR}/aggregated.json")

    # ---- LaTeX Table ----
    print(f"\n--- LaTeX Table (copy to paper) ---")
    print(r"\begin{tabular}{@{}lccccccc@{}}")
    print(r"\toprule")
    print(r"Architecture & Type & $L=2$ & $L=3$ & $L=4$ & $L=6$ & $L=8$ & $L=10$ \\")
    print(r"\midrule")

    # Type-A first
    for arch in archs:
        if TYPE_LABELS.get(arch, '?') == 'A':
            _print_latex_row(arch, agg, all_results)
    print(r"\midrule")
    for arch in archs:
        if TYPE_LABELS.get(arch, '?') == 'B':
            _print_latex_row(arch, agg, all_results)
    print(r"\bottomrule")
    print(r"\end{tabular}")


def _print_latex_row(arch, agg, all_results):
    """Print a LaTeX table row for one architecture."""
    nice_names = {
        'klein_cover': 'Cover',
        'klein_transport': 'Transport',
        'klein_homotopy': 'Homotopy',
        'klein_transformer': 'Transformer',
        'klein_transformer_wc': 'Transf.\\ (WC)',
    }
    name = nice_names.get(arch, arch)
    typ = TYPE_LABELS.get(arch, '?')

    cells = []
    for L in TEST_LENGTHS:
        data = agg.get(arch, {}).get(L)
        if data:
            m, s = data['mean'], data['std']
            if L == 10:
                cells.append(f"$\\mathbf{{{m:.2f}{{\\scriptstyle\\pm{s:.2f}}}}}$")
            else:
                cells.append(f"${m:.2f}{{\\scriptstyle\\pm{s:.2f}}}$")
        else:
            cells.append("---")

    row = f"{name:<20} & {typ} & " + " & ".join(cells) + r" \\"
    print(row)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-seed Klein bottle experiment")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_points', type=int, default=32)
    parser.add_argument('--n_avg', type=int, default=30)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--archs', type=str, nargs='+', default=None)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()

    if args.quick:
        args.epochs, args.n_samples, args.n_avg = 50, 50, 10
    elif args.full:
        args.epochs, args.n_samples, args.n_avg = 500, 1000, 50

    if args.aggregate_only:
        all_results = []
        for seed in SEEDS:
            json_path = os.path.join(RESULTS_DIR, f'seed_{seed}', f'klein_seed{seed}.json')
            if os.path.exists(json_path):
                with open(json_path) as f:
                    all_results.append(json.load(f))
                print(f"Loaded seed {seed} results")
            else:
                print(f"WARNING: missing {json_path}")

        if len(all_results) >= 2:
            aggregate_results(all_results)
        else:
            print("ERROR: need at least 2 seed results to aggregate")
        return

    # Run all seeds
    all_results = []
    for seed in SEEDS:
        result = run_single_seed(seed, args)
        if result is not None:
            all_results.append(result)

    if len(all_results) >= 2:
        aggregate_results(all_results)
    else:
        print(f"ERROR: only {len(all_results)}/{len(SEEDS)} seeds completed")


if __name__ == '__main__':
    main()
