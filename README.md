# Functorial Neural Architectures from Higher Inductive Types

Code and formalization for the paper *Functorial Neural Architectures from Higher Inductive Types*.

## Repository structure

```
hott_neuro/
├── agda/                        # Cubical Agda formalization
│   ├── Torus.agda               # T² as HIT; transport commutativity
│   ├── WedgeOfCircles.agda      # S¹∨S¹; ab ≠ ba in F₂
│   ├── TransportCoherence.agda  # Theorem 3.3: transport decoder is a strict monoidal functor
│   └── NonCompositionality.agda # Theorem 4.1: transport coherence + global mixing → ⊥
├── python/                      # Experiment code
│   ├── torus_experiment.py      # Experiment 1: T² (§5.1, Table 2)
│   ├── wedge_experiment.py      # Experiment 2: S¹∨S¹ (§5.2, Table 3)
│   ├── klein_experiment.py      # Experiment 3: Klein bottle (§5.3, Table 4)
│   ├── run_torus_multiseed.py   # Multi-seed runner for T² (3 seeds, aggregation)
│   └── run_klein_multiseed.py   # Multi-seed runner for Klein bottle (3 seeds, aggregation)
├── requirements.txt
└── README.md
```

## Requirements

**Python experiments:**
- Python 3.10+
- PyTorch 2.0+, NumPy 1.24+, Matplotlib 3.7+

```bash
pip install -r requirements.txt
```

**Agda formalization:**
- Agda 2.6.4+ with `--cubical --safe`
- [agda/cubical](https://github.com/agda/cubical) library

## Reproducing the experiments

All experiments train on words of length ≤ 2 and test on lengths 3, 4, 6, 8, 10.  Results are reported as per-segment Chamfer distance (mean ± std over 3 seeds).

### Experiment 1: Torus T² (Table 2)

```bash
cd python
# Single seed (quick test)
python torus_experiment.py --quick --device cuda

# Full 3-seed run (reproduces Table 2)
python run_torus_multiseed.py --full --device cuda
```

Architectures: Transformer (WC), Cover, Transport Attention, Transport, Homotopy.

### Experiment 2: Wedge of circles S¹∨S¹ (Table 3)

```bash
cd python
# Full 3-seed run (reproduces Table 3)
python wedge_experiment.py --full --n_seeds 3 --device cuda

# With matched-loss ablation (reproduces Appendix C)
python wedge_experiment.py --full --n_seeds 3 --ablation --device cuda
```

Architectures: Transformer, Sequential (GRU), Transport.

### Experiment 3: Klein bottle K (Table 4)

```bash
cd python
# Single seed (quick test)
python klein_experiment.py --quick --device cuda

# Full 3-seed run (reproduces Table 4)
python run_klein_multiseed.py --full --device cuda
```

Architectures: Cover, Transformer (WC), Transport, Homotopy.

Results include canonical/non-canonical word split.

## Checking the Agda formalization

```bash
cd agda
# Each file type-checks independently (requires agda/cubical on the path)
agda Torus.agda
agda WedgeOfCircles.agda
agda TransportCoherence.agda
agda NonCompositionality.agda
```

All modules use `--cubical --safe` with no postulates.

### What is formalized

| Module | Paper result | What is proved |
|--------|-------------|----------------|
| `Torus.agda` | §2.1, Theorem 3.3 | T² as HIT; transport commutativity holds definitionally; winding additivity |
| `WedgeOfCircles.agda` | §2.1, §5.2 | ab ≠ ba in F₂; concatenation is a free monoid |
| `TransportCoherence.agda` | Theorem 3.3 | Transport decoder is a strict monoidal functor (by induction on first word) |
| `NonCompositionality.agda` | Theorem 4.1, Appendix F | Transport coherence and global mixing are contradictory |

### What is not formalized

- The softmax instantiation (that softmax outputs are strictly positive, making attention globally mixing) relies on standard real analysis, not formalized.
- The Klein bottle relation and proof term H are not yet formalized.
- The depth lower bound (Barrington's theorem) is stated as a comment, not a postulate, to preserve `--safe`.

## Paper–code correspondence

| Paper table/figure | Script | Command |
|---|---|---|
| Table 2 (T², all architectures) | `run_torus_multiseed.py` | `--full --device cuda` |
| Table 3 (S¹∨S¹, Chamfer + circle acc.) | `wedge_experiment.py` | `--full --n_seeds 3` |
| Table 4 (Klein, canonical/non-canonical) | `run_klein_multiseed.py` | `--full --device cuda` |
| Appendix C (matched-loss ablation) | `wedge_experiment.py` | `--full --n_seeds 3 --ablation` |
| Appendix D (full L-progression, S¹∨S¹) | `wedge_experiment.py` | (included in `--full` output) |
| Appendix D (full L-progression, Klein) | `run_klein_multiseed.py` | (included in `--full` output) |
| Appendix E (coherence battery, T²) | `torus_experiment.py` | (included in `--full` output) |
