#!/usr/bin/env python3
"""
Multi-seed T² experiment runner.
Runs torus_experiment.py with 3 seeds, then aggregates results into
mean±std tables matching the S¹∨S¹ paper format.

Usage:
    # Full run (matches paper settings: 500 epochs, 1000 samples)
    python run_torus_multiseed.py --full --device cuda

    # Quick sanity check
    python run_torus_multiseed.py --quick --device cuda

    # Custom
    python run_torus_multiseed.py --epochs 300 --n_samples 500 --device cuda
"""

import json
import os
import sys
import subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict

SEEDS = [42, 179, 316]  # same seeds as S¹∨S¹ experiment
SCRIPT = os.path.join(os.path.dirname(__file__), 'torus_experiment.py')
RESULTS_DIR = 'results_torus_multiseed'
TEST_LENGTHS = [2, 3, 4, 6, 8, 10]


def run_single_seed(seed, args):
    """Run one seed of the T² experiment."""
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
    print(f"  SEED {seed}  ({SEEDS.index(seed)+1}/{len(SEEDS)})")
    print(f"{'='*70}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: seed {seed} failed with return code {result.returncode}")
        return None

    json_path = os.path.join(output_dir, f'torus_seed{seed}.json')
    if not os.path.exists(json_path):
        print(f"ERROR: expected output not found: {json_path}")
        return None

    with open(json_path) as f:
        return json.load(f)


def aggregate_results(all_results):
    """Aggregate per-seed results into mean±std tables."""
    # Collect all architecture names
    archs = list(all_results[0].keys())

    # ---- Training loss ----
    print(f"\n{'='*80}")
    print("AGGREGATED T² RESULTS (mean ± std over 3 seeds)")
    print(f"{'='*80}")

    print(f"\n--- Training Loss ---")
    print(f"{'Architecture':<20} {'Mean':>10} {'Std':>10} {'Seeds':>20}")
    print("-" * 60)
    for arch in archs:
        losses = [r[arch]['final_loss'] for r in all_results if arch in r]
        print(f"{arch:<20} {np.mean(losses):10.4f} {np.std(losses):10.4f} {str([f'{l:.4f}' for l in losses]):>20}")

    # ---- Per-segment Chamfer (THE key table) ----
    print(f"\n--- Per-Segment Chamfer Distance (mean ± std) ---")
    header = f"{'Architecture':<20} {'Type':>6}"
    for L in TEST_LENGTHS:
        header += f" {'L='+str(L):>12}"
    print(header)
    print("-" * (26 + 13 * len(TEST_LENGTHS)))

    type_map = {
        'transformer': 'A', 'transformer_wc': 'A',
        'torus_cover': 'A', 'transport_attn': 'A',
        'torus_transport': 'B', 'torus_homotopy': 'B',
    }

    per_seg_agg = {}
    for arch in archs:
        row = f"{arch:<20} {type_map.get(arch, '?'):>6}"
        per_seg_agg[arch] = {}
        for L in TEST_LENGTHS:
            vals = []
            for r in all_results:
                if arch not in r:
                    continue
                t5 = r[arch].get('battery', {}).get('t5_scaling', {})
                key = str(L) if str(L) in t5 else L
                if key in t5 and 'per_seg_chamfer' in t5[key]:
                    vals.append(t5[key]['per_seg_chamfer'])
            if vals:
                m, s = np.mean(vals), np.std(vals)
                per_seg_agg[arch][L] = (m, s)
                row += f" {m:.3f}±{s:.3f}"
            else:
                row += f" {'—':>12}"
        print(row)

    # ---- Upsampled Chamfer ----
    print(f"\n--- Upsampled Chamfer Distance (mean ± std) ---")
    header = f"{'Architecture':<20} {'Type':>6}"
    for L in TEST_LENGTHS:
        header += f" {'L='+str(L):>12}"
    print(header)
    print("-" * (26 + 13 * len(TEST_LENGTHS)))

    for arch in archs:
        row = f"{arch:<20} {type_map.get(arch, '?'):>6}"
        for L in TEST_LENGTHS:
            vals = []
            for r in all_results:
                if arch not in r:
                    continue
                t5 = r[arch].get('battery', {}).get('t5_scaling', {})
                key = str(L) if str(L) in t5 else L
                if key in t5:
                    v = t5[key].get('mean_chamfer_up', t5[key].get('mean_chamfer'))
                    if v is not None:
                        vals.append(v)
            if vals:
                row += f" {np.mean(vals):.3f}±{np.std(vals):.3f}"
            else:
                row += f" {'—':>12}"
        print(row)

    # ---- Winding accuracy (angle space) ----
    print(f"\n--- Winding Accuracy, angle-space (mean ± std) ---")
    header = f"{'Architecture':<20} {'Type':>6}"
    for L in TEST_LENGTHS:
        header += f" {'L='+str(L):>12}"
    print(header)
    print("-" * (26 + 13 * len(TEST_LENGTHS)))

    for arch in archs:
        row = f"{arch:<20} {type_map.get(arch, '?'):>6}"
        for L in TEST_LENGTHS:
            vals = []
            for r in all_results:
                if arch not in r:
                    continue
                t5 = r[arch].get('battery', {}).get('t5_scaling', {})
                key = str(L) if str(L) in t5 else L
                if key in t5:
                    v = t5[key].get('winding_angle', t5[key].get('winding_accuracy'))
                    if v is not None:
                        vals.append(v)
            if vals:
                row += f" {np.mean(vals)*100:.0f}±{np.std(vals)*100:.0f}%"
            else:
                row += f" {'—':>12}"
        print(row)

    # ---- Coherence battery ----
    print(f"\n--- Coherence Battery (mean ± std) ---")
    print(f"{'Architecture':<20} {'Comp d∞':>12} {'Comm d∞':>12} {'Reord':>12} {'Noncan':>12}")
    print("-" * 68)

    for arch in archs:
        comp_vals, comm_vals, reord_vals, noncan_vals = [], [], [], []
        for r in all_results:
            if arch not in r:
                continue
            b = r[arch].get('battery', {})
            if 't1' in b:
                v = b['t1'].get('mean', b['t1'].get('mean_frechet', None))
                if v is not None: comp_vals.append(v)
            if 't2' in b:
                v = b['t2'].get('frechet', b['t2'].get('mean_frechet', b['t2'].get('mean', None)))
                if v is not None: comm_vals.append(v)
            if 't3' in b:
                v = b['t3'].get('mean_naive', b['t3'].get('naive', b['t3'].get('mean', None)))
                if v is not None: reord_vals.append(v)
            if 't4' in b:
                v = b['t4'].get('frechet', b['t4'].get('mean_frechet', b['t4'].get('mean', None)))
                if v is not None:
                    noncan_vals.append(v)

        def fmt(vals):
            if vals:
                return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"
            return "—"

        print(f"{arch:<20} {fmt(comp_vals):>12} {fmt(comm_vals):>12} "
              f"{fmt(reord_vals):>12} {fmt(noncan_vals):>12}")

    # ---- Gap summary ----
    print(f"\n--- Gap Summary at L=10 ---")
    type_b_vals = []
    type_a_vals = []
    for arch in archs:
        if 10 in per_seg_agg.get(arch, {}):
            m = per_seg_agg[arch][10][0]
            if type_map.get(arch) == 'B':
                type_b_vals.append((arch, m))
            else:
                type_a_vals.append((arch, m))

    if type_b_vals and type_a_vals:
        b_range = f"{min(v for _, v in type_b_vals):.3f}–{max(v for _, v in type_b_vals):.3f}"
        a_range = f"{min(v for _, v in type_a_vals):.3f}–{max(v for _, v in type_a_vals):.3f}"
        b_mid = np.mean([v for _, v in type_b_vals])
        a_mid = np.mean([v for _, v in type_a_vals])
        print(f"  Type-B: {b_range}")
        print(f"  Type-A: {a_range}")
        print(f"  Gap ratio: {a_mid/b_mid:.1f}×")

    # ---- Save aggregated JSON ----
    agg_path = os.path.join(RESULTS_DIR, 'aggregated.json')
    agg = {}
    for arch in archs:
        agg[arch] = {
            'type': type_map.get(arch, '?'),
            'training_loss': {
                'mean': float(np.mean([r[arch]['final_loss'] for r in all_results if arch in r])),
                'std': float(np.std([r[arch]['final_loss'] for r in all_results if arch in r])),
                'per_seed': [r[arch]['final_loss'] for r in all_results if arch in r],
            },
            'per_seg_chamfer': {
                str(L): {'mean': float(m), 'std': float(s)}
                for L, (m, s) in per_seg_agg.get(arch, {}).items()
            },
        }
    with open(agg_path, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\nAggregated results saved to: {agg_path}")

    # ---- LaTeX table for paper ----
    print(f"\n--- LaTeX Table (copy to paper) ---")
    print(r"\begin{tabular}{@{}lccccccc@{}}")
    print(r"\toprule")
    print(r"Architecture & Type & $L=2$ & $L=3$ & $L=4$ & $L=6$ & $L=8$ & $L=10$ \\")
    print(r"\midrule")

    for arch in sorted(archs, key=lambda a: (0 if type_map.get(a)=='A' else 1, a)):
        nice = arch.replace('torus_', '').replace('_', ' ').title()
        nice = nice.replace('Transformer Wc', 'Transformer (WC)')
        nice = nice.replace('Transport Attn', 'Transport Attn.')
        row = f"{nice:<20} & {type_map.get(arch, '?')}"
        for L in TEST_LENGTHS:
            if L in per_seg_agg.get(arch, {}):
                m, s = per_seg_agg[arch][L]
                cell = f"${m:.2f} \\pm {s:.2f}$"
                if L == 10:
                    cell = f"$\\mathbf{{{m:.2f} \\pm {s:.2f}}}$"
                row += f" & {cell}"
            else:
                row += " & —"
        row += r" \\"
        if arch == sorted([a for a in archs if type_map.get(a) == 'A'],
                          key=lambda a: a)[-1]:
            row += "\n\\midrule"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-seed T² experiment (3 seeds, aggregated)")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_points', type=int, default=32)
    parser.add_argument('--n_avg', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--archs', type=str, nargs='+', default=None)
    parser.add_argument('--quick', action='store_true',
                        help='Fast sanity check (50 epochs, 50 samples)')
    parser.add_argument('--full', action='store_true',
                        help='Full paper settings (500 epochs, 1000 samples)')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS,
                        help='Seeds to run (default: 42 179 316)')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Skip training, just aggregate existing results')
    args = parser.parse_args()

    if args.quick:
        args.epochs, args.n_samples, args.n_avg = 50, 50, 10
    elif args.full:
        args.epochs, args.n_samples, args.n_avg = 500, 1000, 50

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not args.aggregate_only:
        for seed in args.seeds:
            run_single_seed(seed, args)

    # Aggregate
    all_results = []
    for seed in args.seeds:
        output_dir = os.path.join(RESULTS_DIR, f'seed_{seed}')
        json_path = os.path.join(output_dir, f'torus_seed{seed}.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                all_results.append(json.load(f))
            print(f"Loaded seed {seed} results")
        else:
            print(f"WARNING: missing results for seed {seed} at {json_path}")

    if len(all_results) < 2:
        print(f"ERROR: need at least 2 seeds, found {len(all_results)}")
        sys.exit(1)

    aggregate_results(all_results)


if __name__ == '__main__':
    main()
