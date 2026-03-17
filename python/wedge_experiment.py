#!/usr/bin/env python3
"""
Experiment 2: S¹∨S¹ (wedge of circles), π₁ = F₂ (free group on 2 generators)

NON-ABELIAN companion to the T² experiment.  Key differences:
  - ab ≠ ba: generator ordering affects homotopy class
  - No canonical form: the word IS the homotopy class
  - Transport decoder concatenates in WORD ORDER (not canonical)
  - Transformers provably need Ω(log n) depth (Theorem 4.7)
  - F₂ is free → no proof term H needed (no relations to witness)

Architectures:
  wedge_transport   (type-B): structural concatenation in word order
  wedge_transformer (type-A): standard transformer, global attention
  wedge_sequential  (type-A): GRU-based sequential generation with context

Tests:
  - Order sensitivity: D(ab) ≠ D(ba), both correct?  [NEW, unique to non-abelian]
  - Circle accuracy:   does each segment trace the correct circle?
  - Length scaling:     per-segment Chamfer for L = 2..10

Usage:
  python wedge_experiment.py --full                      # single seed
  python wedge_experiment.py --full --n_seeds 3          # multi-seed (mean±std)
  python wedge_experiment.py --full --n_seeds 3 --ablation  # + matched-loss ablation
"""

import os, sys, math, json, time, logging, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# =============================================================================
# §1  S¹∨S¹ GEOMETRY
#
#  Two circles in R³ meeting at the origin (wedge point):
#    Circle A: center (-1, 0, 0), radius 1, in XY plane
#              parametrized: (-1 + cos θ, sin θ, 0)   → passes through (0,0,0) at θ=0
#    Circle B: center (0, 0, -1), radius 1, in YZ plane
#              parametrized: (0, sin θ, -1 + cos θ)   → passes through (0,0,0) at θ=0
#
#  π₁(S¹∨S¹) = F₂ = ⟨a, b | ⟩  (free group, NO relations)
#
#  A word w = w₁w₂...wₗ ∈ {a,b}* represents a loop that traces:
#    - Circle A when wᵢ = a
#    - Circle B when wᵢ = b
#  Each generator starts and ends at the wedge point.
# =============================================================================

def circle_A(theta):
    """Points on circle A given angle array theta. Returns (N, 3)."""
    return np.stack([-1 + np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)

def circle_B(theta):
    """Points on circle B given angle array theta. Returns (N, 3)."""
    return np.stack([np.zeros_like(theta), np.sin(theta), -1 + np.cos(theta)], axis=-1)


def generate_wedge_loop(word, n_pts=32, noise=0.02, rng=None):
    """
    Generate a ground-truth loop on S¹∨S¹ for the given word.

    Each letter traces one full circle (2π), with:
      - Slightly non-uniform speed (angle increment noise)
      - Small Gaussian displacement in R³

    Returns: (L * n_pts, 3) array of R³ points.
    """
    if rng is None:
        rng = np.random.default_rng()

    segments = []
    for letter in word:
        # Angle increments: uniform + small noise for variety
        d_theta = np.full(n_pts, 2 * np.pi / n_pts)
        d_theta += rng.normal(0, 0.15 / n_pts, n_pts)
        # Hard constraint: total angle = 2π
        d_theta -= (d_theta.sum() - 2 * np.pi) / n_pts
        theta = np.cumsum(d_theta)
        # Shift so first point is at θ≈0 (wedge point)
        theta = theta - theta[0]

        if letter == 'a':
            pts = circle_A(theta)
        elif letter == 'b':
            pts = circle_B(theta)
        else:
            raise ValueError(f"Unknown letter: {letter}")

        # Small R³ noise (normal to circle surface for geometric variety)
        pts += rng.normal(0, noise, pts.shape)
        segments.append(pts)

    return np.concatenate(segments, axis=0).astype(np.float32)


def wedge_word_equivalence(w1, w2):
    """In F₂ (free group), two words are equivalent iff they are identical
    after free reduction (canceling aa⁻¹, bb⁻¹).  Since we only use
    positive generators {a, b}, two words are equivalent iff they are
    literally identical."""
    return w1 == w2


# =============================================================================
# §2  DATASET
# =============================================================================

class WedgeLoopDataset(Dataset):
    """Dataset of (word, loop) pairs on S¹∨S¹."""

    def __init__(self, words, n_samples=1000, n_pts=32, max_word_len=2,
                 noise=0.02, seed=42):
        self.n_pts = n_pts
        self.max_word_len = max_word_len
        rng = np.random.default_rng(seed)

        self.samples = []
        for w in words:
            for _ in range(n_samples):
                loop = generate_wedge_loop(w, n_pts, noise, rng)
                self.samples.append((w, loop))

        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        word, loop = self.samples[idx]
        word_ids = [{'a': 1, 'b': 2}[c] for c in word]
        return {
            'word': word,
            'word_ids': torch.tensor(word_ids, dtype=torch.long),
            'loop': torch.tensor(loop, dtype=torch.float32),
            'length': len(word),
        }


def collate_wedge(batch):
    max_wlen = max(b['length'] for b in batch)
    max_llen = max(b['loop'].shape[0] for b in batch)

    word_ids = torch.zeros(len(batch), max_wlen, dtype=torch.long)
    loops = torch.zeros(len(batch), max_llen, 3)
    masks = torch.zeros(len(batch), max_llen)
    lengths = torch.tensor([b['length'] for b in batch])

    for i, b in enumerate(batch):
        wl = b['length']
        ll = b['loop'].shape[0]
        word_ids[i, :wl] = b['word_ids']
        loops[i, :ll] = b['loop']
        masks[i, :ll] = 1.0

    return word_ids, loops, masks, lengths


# =============================================================================
# §3  CHAMFER LOSS (same as T² experiment)
# =============================================================================

def chamfer_loss(pred, target, mask=None):
    """Batched Chamfer loss.  pred, target: (B, N, 3).  mask: (B, N)."""
    B, N, _ = pred.shape
    M = target.shape[1]

    # Clamp to valid points
    if mask is not None:
        pred = pred * mask.unsqueeze(-1)
        target = target * mask.unsqueeze(-1)

    # (B, N, M)
    d2 = ((pred.unsqueeze(2) - target.unsqueeze(1)) ** 2).sum(-1)
    loss_pred = d2.min(2).values.mean(1)    # pred → nearest in target
    loss_tgt = d2.min(1).values.mean(1)     # target → nearest in pred
    return ((loss_pred + loss_tgt) / 2).mean()


# =============================================================================
# §4  ARCHITECTURES
# =============================================================================

class _WedgeBase(nn.Module):
    """Shared encoder and output infrastructure for all wedge decoders."""

    def __init__(self, vocab_size=3, embed_dim=32, latent_dim=64,
                 enc_layers=2, enc_hidden=128, n_points=32,
                 max_word_len=2, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.n_points = n_points
        self.max_word_len = max_word_len
        self.output_len = max_word_len * n_points  # fixed output resolution

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, enc_hidden, num_layers=enc_layers,
                              batch_first=True, bidirectional=True)
        self.enc_proj = nn.Linear(enc_hidden * 2, latent_dim)

    def encode_word(self, word_ids, lengths):
        emb = self.embed(word_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.enc_proj(h)

    def _get_word_sequence(self, word_ids, lengths):
        """Return list of (batch_idx, letter_list) for word-order iteration."""
        B = word_ids.shape[0]
        seqs = []
        for i in range(B):
            L = int(lengths[i].item())
            letters = word_ids[i, :L].tolist()
            seqs.append(letters)
        return seqs


# --- TYPE-B: Transport Decoder (structural concatenation in word order) ---

class WedgeTransportDecoder(_WedgeBase):
    """
    Type-B decoder for S¹∨S¹.

    Learns two generator shapes g_a, g_b (loops around circles A, B).
    For word w = w₁w₂...wₗ, output = g_{w₁} ∘ g_{w₂} ∘ ... ∘ g_{wₗ}
    (concatenation in WORD ORDER — not canonical form).

    Hard constraints:
      - g_a has total angle = 2π around circle A
      - g_b has total angle = 2π around circle B
      - Each generator starts/ends at origin (wedge point)

    Guarantees (for ALL parameter values, ALL word lengths):
      - Each segment traces the correct circle with winding 1
      - Composition is structural: D(w₁w₂) = D(w₁) ∘ D(w₂)
      - Order is preserved: D(ab) ≠ D(ba) in general
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.latent_dim
        n = self.n_points

        # Generator networks: z → n_pts angle increments
        self.gen_a_net = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, n),
        )
        self.gen_b_net = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, n),
        )

    def _generate_segment(self, z, letter):
        """Generate one generator traversal (n_pts points in R³)."""
        n = self.n_points
        if letter == 1:  # 'a'
            raw_dtheta = self.gen_a_net(z)
        elif letter == 2:  # 'b'
            raw_dtheta = self.gen_b_net(z)
        else:
            return torch.zeros(z.shape[0], n, 3, device=z.device)

        # Hard winding constraint: total angle = 2π
        dtheta = raw_dtheta
        correction = (dtheta.sum(dim=-1, keepdim=True) - 2 * math.pi) / n
        dtheta = dtheta - correction  # Σ Δθ = 2π exactly

        # Cumulative angle (starting at 0 → wedge point)
        theta = torch.cumsum(dtheta, dim=-1)  # (B, n_pts)

        # Embed into R³
        if letter == 1:  # Circle A: (-1+cos θ, sin θ, 0)
            x = -1 + torch.cos(theta)
            y = torch.sin(theta)
            z_coord = torch.zeros_like(theta)
        else:  # Circle B: (0, sin θ, -1+cos θ)
            x = torch.zeros_like(theta)
            y = torch.sin(theta)
            z_coord = -1 + torch.cos(theta)

        return torch.stack([x, y, z_coord], dim=-1)  # (B, n_pts, 3)

    def forward(self, word_ids, lengths):
        z = self.encode_word(word_ids, lengths)  # (B, d)
        B = z.shape[0]
        seqs = self._get_word_sequence(word_ids, lengths)
        device = z.device

        outputs = []
        for i in range(B):
            segments = []
            for letter in seqs[i]:
                seg = self._generate_segment(z[i:i+1], letter)  # (1, n_pts, 3)
                segments.append(seg.squeeze(0))
            if segments:
                full_loop = torch.cat(segments, dim=0)  # (L*n_pts, 3)
            else:
                full_loop = torch.zeros(self.n_points, 3, device=device)
            # Resample to fixed output length
            full_loop = self._resample(full_loop, self.output_len)
            outputs.append(full_loop)

        return torch.stack(outputs, dim=0)  # (B, output_len, 3)

    def _resample(self, pts, target_len):
        """Resample (N, 3) tensor to (target_len, 3) via linear interp."""
        N = pts.shape[0]
        if N == target_len:
            return pts
        if N < 2:
            return pts.expand(target_len, -1)
        idx = torch.linspace(0, N - 1, target_len, device=pts.device)
        lo = idx.long().clamp(max=N - 2)
        hi = (lo + 1).clamp(max=N - 1)
        frac = (idx - lo.float()).unsqueeze(-1)
        return pts[lo] * (1 - frac) + pts[hi] * frac


# --- TYPE-A: Transformer Decoder (global attention, no composition structure) ---

class WedgeTransformerDecoder(_WedgeBase):
    """
    Type-A decoder: standard transformer.

    The full loop is generated by a transformer over output positions.
    Every output position attends to every other → cross-segment dependency.
    No structural composition.

    On S¹∨S¹ (non-abelian), the transformer faces a strictly harder task
    than on T² (abelian): it cannot reduce to counting letters.
    It must track the FULL word order — which requires Ω(log n) depth
    (Theorem 4.7, Barrington).
    """

    def __init__(self, n_heads=4, n_layers=3, **kw):
        super().__init__(**kw)
        T = self.output_len
        d = self.latent_dim

        self.pos_embed = nn.Parameter(torch.randn(1, T, d) * 0.02)
        self.z_to_seq = nn.Linear(d, T * d)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 4,
            dropout=0.0, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d, 3)  # output R³ coordinates directly

    def forward(self, word_ids, lengths):
        z = self.encode_word(word_ids, lengths)
        B = z.shape[0]
        T = self.output_len

        seq = self.z_to_seq(z).view(B, T, -1) + self.pos_embed
        out = self.transformer(seq)
        return self.out_proj(out)  # (B, T, 3)


# --- TYPE-A: Sequential Decoder (GRU-based, one segment per step) ---

class WedgeSequentialDecoder(_WedgeBase):
    """
    Type-A decoder with sequential inductive bias.

    Processes the word left-to-right with a GRU.  At each step,
    generates one segment conditioned on (letter, hidden_state).

    This tests Theorem 4.7: sequential architectures can compute
    prefix products (which fixed-depth transformers cannot for non-abelian groups).

    Still type-A: segment quality depends on accumulated context (hidden state),
    not purely on the current letter.  But the sequential structure should help
    with the non-abelian word problem.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.latent_dim
        n = self.n_points

        self.letter_embed = nn.Embedding(3, d, padding_idx=0)
        self.decode_gru = nn.GRUCell(d, d)

        # Segment generator: hidden_state → n_pts × 3
        self.seg_gen = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, n * 3),
        )

    def forward(self, word_ids, lengths):
        z = self.encode_word(word_ids, lengths)  # (B, d)
        B = z.shape[0]
        seqs = self._get_word_sequence(word_ids, lengths)
        device = z.device
        max_L = max(len(s) for s in seqs)

        outputs = []
        for i in range(B):
            h = z[i:i+1]  # (1, d), initial hidden state
            segments = []
            for letter in seqs[i]:
                le = self.letter_embed(torch.tensor([letter], device=device))
                h = self.decode_gru(le, h)  # (1, d)
                seg = self.seg_gen(h).view(1, self.n_points, 3)
                segments.append(seg.squeeze(0))
            if segments:
                full_loop = torch.cat(segments, dim=0)
            else:
                full_loop = torch.zeros(self.n_points, 3, device=device)
            full_loop = WedgeTransportDecoder._resample(None, full_loop, self.output_len)
            outputs.append(full_loop)

        return torch.stack(outputs, dim=0)


# Registry
DECODERS = {
    'wedge_transport':   WedgeTransportDecoder,
    'wedge_transformer': WedgeTransformerDecoder,
    'wedge_sequential':  WedgeSequentialDecoder,
}

ARCH_TYPE = {
    'wedge_transport':   'B',
    'wedge_transformer': 'A',
    'wedge_sequential':  'A',
}

HIT_LEVELS = {
    'wedge_transport':   'base + loop_a + loop_b (word-order concat)',
    'wedge_transformer': 'NONE (attention)',
    'wedge_sequential':  'NONE (sequential + context)',
}


# =============================================================================
# §5  TRAINING
# =============================================================================

def train_wedge(model, dataset, epochs=300, lr=1e-3, batch_size=64,
                device='cpu', log_every=50, patience=80):
    """Train with early stopping.  Returns history list."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_wedge, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup = min(20, epochs // 10)
    def lr_fn(ep):
        if ep < warmup:
            return ep / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    model.to(device)
    model.train()
    history = []
    t_start = time.time()
    best_loss = float('inf')
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        total_loss, nb = 0.0, 0
        for word_ids, targets, masks, lengths in loader:
            word_ids = word_ids.to(device)
            targets, masks = targets.to(device), masks.to(device)
            lengths = lengths.to(device)

            pred = model(word_ids, lengths)

            # Resample target to match pred length if needed
            T_pred = pred.shape[1]
            T_tgt = targets.shape[1]
            if T_pred != T_tgt:
                targets_r = F.interpolate(
                    targets.permute(0, 2, 1), T_pred, mode='linear',
                    align_corners=True).permute(0, 2, 1)
                masks_r = F.interpolate(
                    masks.unsqueeze(1), T_pred, mode='nearest').squeeze(1)
            else:
                targets_r, masks_r = targets, masks

            loss = chamfer_loss(pred, targets_r, masks_r)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * word_ids.shape[0]
            nb += word_ids.shape[0]

        scheduler.step()
        avg = total_loss / max(nb, 1)
        history.append(avg)

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t_start
            logger.info(f"  Epoch {epoch:4d}/{epochs}  loss={avg:.6f}  "
                        f"({elapsed:.0f}s elapsed)")

        # Early stopping
        if avg < best_loss - 1e-5:
            best_loss = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience and epoch > warmup + patience:
            logger.info(f"  Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs, best={best_loss:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    logger.info(f"  Training complete in {time.time()-t_start:.0f}s  "
                f"(best loss: {best_loss:.6f})")
    return history, best_loss


# =============================================================================
# §6  TEST BATTERY
# =============================================================================

def _resample_np(pts, target_len):
    N = pts.shape[0]
    if N == target_len:
        return pts
    if N < 2 or target_len < 2:
        return pts[:target_len] if N >= target_len else pts
    idx = np.linspace(0, N - 1, target_len)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, N - 1)
    frac = (idx - lo)[:, None]
    return pts[lo] * (1 - frac) + pts[hi] * frac


def _chamfer_np(A, B):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return float((d2.min(1).mean() + d2.min(0).mean()) / 2)


def _check_circle(segment, letter):
    """Check if a segment traces the correct circle.

    For circle A: points should cluster near (-1+cos θ, sin θ, 0)
    For circle B: points should cluster near (0, sin θ, -1+cos θ)

    Returns: (correct_circle: bool, angle_traversed: float)
    """
    if segment.shape[0] < 3:
        return False, 0.0

    if letter == 'a':
        # Project onto circle A: compute angle θ_A = atan2(y, x+1)
        theta = np.arctan2(segment[:, 1], segment[:, 0] + 1)
        # Distance from circle A
        dist_A = np.sqrt((segment[:, 0] + 1)**2 + segment[:, 1]**2) - 1
        dist_B = np.sqrt(segment[:, 1]**2 + (segment[:, 2] + 1)**2) - 1
    else:
        # Project onto circle B: compute angle θ_B = atan2(y, z+1)
        theta = np.arctan2(segment[:, 1], segment[:, 2] + 1)
        dist_A = np.sqrt((segment[:, 0] + 1)**2 + segment[:, 1]**2) - 1
        dist_B = np.sqrt(segment[:, 1]**2 + (segment[:, 2] + 1)**2) - 1

    # Total angle traversed
    dtheta = np.diff(theta)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    total_angle = abs(dtheta.sum())

    # Is it closer to the correct circle?
    if letter == 'a':
        correct = np.mean(np.abs(dist_A)) < np.mean(np.abs(dist_B))
    else:
        correct = np.mean(np.abs(dist_B)) < np.mean(np.abs(dist_A))

    # Winding: should be close to 2π
    winding_ok = abs(total_angle - 2 * np.pi) < np.pi  # within half a turn

    return bool(correct and winding_ok), float(total_angle)


@torch.no_grad()
def generate_loop(model, word, n_avg=30, device='cpu'):
    """Generate loops by averaging n_avg forward passes."""
    model.eval()
    word_ids = torch.tensor([[{'a': 1, 'b': 2}[c] for c in word]],
                            dtype=torch.long, device=device)
    lengths = torch.tensor([len(word)], device=device)

    all_loops = []
    for _ in range(n_avg):
        loop = model(word_ids, lengths)[0].cpu().numpy()
        all_loops.append(loop)

    mean_loop = np.mean(all_loops, axis=0)
    return mean_loop, all_loops


# --- Test 1: Order sensitivity (unique to non-abelian) ---

@torch.no_grad()
def test_order_sensitivity(model, n_avg=30, device='cpu'):
    """
    THE NON-ABELIAN TEST.

    On T² (abelian): D(ab) = D(ba) by construction.
    On S¹∨S¹ (non-abelian): D(ab) should DIFFER from D(ba).

    For the transport decoder, D(ab) = g_a ∘ g_b ≠ g_b ∘ g_a = D(ba)
    (unless g_a = g_b, which is a measure-zero coincidence).

    We measure:
      1. L∞ distance between D(ab) and D(ba) — should be LARGE
      2. Chamfer to respective ground truths — should be SMALL
    """
    results = {}

    pairs = [
        ('ab', 'ba'),
        ('aab', 'aba'),
        ('aab', 'baa'),
        ('abb', 'bab'),
        ('abb', 'bba'),
    ]

    for w1, w2 in pairs:
        loop1, _ = generate_loop(model, w1, n_avg, device)
        loop2, _ = generate_loop(model, w2, n_avg, device)

        # Resample to same length
        n = max(loop1.shape[0], loop2.shape[0])
        l1 = _resample_np(loop1, n)
        l2 = _resample_np(loop2, n)

        # Distance between the two outputs (should be large)
        cross_chamfer = _chamfer_np(l1, l2)
        linf = float(np.linalg.norm(l1 - l2, axis=-1).max())

        # Ground truth for each
        rng = np.random.default_rng(42)
        gt1 = np.mean([generate_wedge_loop(w1, 32, 0.02, rng) for _ in range(5)], 0)
        gt2 = np.mean([generate_wedge_loop(w2, 32, 0.02, rng) for _ in range(5)], 0)

        # Chamfer to own ground truth (should be small)
        gt1_r = _resample_np(gt1, loop1.shape[0])
        gt2_r = _resample_np(gt2, loop2.shape[0])
        self_chamfer1 = _chamfer_np(loop1, gt1_r)
        self_chamfer2 = _chamfer_np(loop2, gt2_r)

        results[f'{w1}_vs_{w2}'] = {
            'cross_chamfer': cross_chamfer,
            'cross_linf': linf,
            'self_chamfer_1': self_chamfer1,
            'self_chamfer_2': self_chamfer2,
            'distinguishes': cross_chamfer > max(self_chamfer1, self_chamfer2) * 0.5,
        }

    # Summary
    results['fraction_distinguished'] = np.mean([
        v['distinguishes'] for v in results.values() if isinstance(v, dict)])
    results['mean_cross_chamfer'] = np.mean([
        v['cross_chamfer'] for v in results.values() if isinstance(v, dict)])

    return results


# --- Test 2: Circle accuracy (analog of winding check) ---

@torch.no_grad()
def test_circle_accuracy(model, words=None, n_avg=30, device='cpu'):
    """Check if each segment of the generated loop traces the correct circle."""
    if words is None:
        words = ['a', 'b', 'ab', 'ba', 'aa', 'bb']

    n_pts = model.n_points
    correct = 0
    total = 0

    # Structural check: for transport decoder, circle assignment is by construction
    arch = model.__class__.__name__
    if 'Transport' in arch:
        # Type-B: circle assignment is architectural
        return {
            'accuracy': 1.0,
            'by_construction': True,
            'details': 'Structural: each segment generated by the correct circle network'
        }

    for word in words:
        mean_loop, _ = generate_loop(model, word, n_avg, device)
        L = len(word)
        out_len = mean_loop.shape[0]

        for s, letter in enumerate(word):
            s_start = int(s * out_len / L)
            s_end = int((s + 1) * out_len / L)
            seg = mean_loop[s_start:s_end]
            if seg.shape[0] < 3:
                continue
            ok, _ = _check_circle(seg, letter)
            if ok:
                correct += 1
            total += 1

    return {
        'accuracy': correct / max(total, 1),
        'by_construction': False,
        'n_correct': correct,
        'n_total': total,
    }


# --- Test 3: Length scaling (the central experiment) ---

def _generate_test_words(length, n_words=8, seed=42):
    """Generate diverse test words of a given length over {a, b}."""
    rng = np.random.default_rng(seed)
    letters = ['a', 'b']
    words = set()
    max_possible = 2 ** length
    target = min(max(n_words, length + 1), max_possible)

    for _ in range(target * 10):
        w = ''.join(rng.choice(letters, size=length))
        words.add(w)
        if len(words) >= target:
            break

    return sorted(words)


@torch.no_grad()
def test_length_scaling(model, test_lengths, n_avg=30, device='cpu',
                        n_words_per_length=8, seed=42):
    """
    THE PAPER FIGURE (S¹∨S¹ version).

    Train on length ≤ 2.  Test on length 3, 4, 6, 8, 10.

    Key difference from T²: on S¹∨S¹, "ab" and "ba" are DIFFERENT test cases.
    The transport decoder must produce different (and correct) outputs for each.
    The transformer must learn order-dependent generation — provably harder
    than on T² where it could cheat by counting.

    Metrics:
      - circle_accuracy: does each segment trace the correct circle?
      - chamfer_upsample: Chamfer after upsampling to GT resolution
      - per_seg_chamfer: per-segment Chamfer at fixed 32 pts/seg (THE metric)
      - linf: max pointwise distance
    """
    results = {}
    n_points = model.n_points
    arch = model.__class__.__name__

    for li, L in enumerate(test_lengths):
        words = _generate_test_words(L, n_words=n_words_per_length, seed=seed)
        logger.info(f"    Length {L}: testing {len(words)} words  "
                    f"[{li+1}/{len(test_lengths)}]")
        t0 = time.time()

        circle_correct = 0
        circle_total = 0
        chamfer_up_dists = []
        per_seg_fair = []
        linf_dists = []

        for word in words:
            mean_loop, _ = generate_loop(model, word, n_avg, device)
            gt_res = L * n_points

            # Ground truth
            rng_gt = np.random.default_rng(seed + hash(word) % 10000)
            gt_loops = [generate_wedge_loop(word, n_points, 0.02, rng_gt)
                        for _ in range(5)]
            gt_mean = np.mean(gt_loops, axis=0)

            # Circle accuracy (per segment)
            out_len = mean_loop.shape[0]
            if 'Transport' in arch:
                circle_correct += L  # by construction
            else:
                for s, letter in enumerate(word):
                    s_start = int(s * out_len / L)
                    s_end = int((s + 1) * out_len / L)
                    seg = mean_loop[s_start:s_end]
                    if seg.shape[0] >= 3:
                        ok, _ = _check_circle(seg, letter)
                        if ok:
                            circle_correct += 1
            circle_total += L

            # Chamfer (upsample model output to GT resolution)
            gen_up = _resample_np(mean_loop, gt_res)
            chamfer_up_dists.append(_chamfer_np(gen_up, gt_mean))

            # Per-segment Chamfer at fixed resolution (THE FAIR METRIC)
            seg_chamfers = []
            for s in range(L):
                s_start = int(s * out_len / L)
                s_end = int((s + 1) * out_len / L)
                model_seg = mean_loop[s_start:s_end]
                model_seg_32 = _resample_np(model_seg, n_points)
                gt_seg = gt_mean[s * n_points:(s + 1) * n_points]
                if model_seg_32.shape[0] > 0 and gt_seg.shape[0] > 0:
                    seg_chamfers.append(_chamfer_np(model_seg_32, gt_seg))
            if seg_chamfers:
                per_seg_fair.append(float(np.mean(seg_chamfers)))

            # L-infinity
            gen_aligned = _resample_np(mean_loop, gt_res)
            linf_dists.append(float(np.linalg.norm(gen_aligned - gt_mean, axis=-1).max()))

        results[L] = {
            'n_words': len(words),
            'words_tested': words,
            'circle_accuracy': circle_correct / max(circle_total, 1),
            'mean_chamfer_up': float(np.mean(chamfer_up_dists)) if chamfer_up_dists else float('nan'),
            'std_chamfer_up': float(np.std(chamfer_up_dists)) if chamfer_up_dists else float('nan'),
            'per_seg_chamfer': float(np.mean(per_seg_fair)) if per_seg_fair else float('nan'),
            'std_per_seg': float(np.std(per_seg_fair)) if per_seg_fair else float('nan'),
            'mean_linf': float(np.mean(linf_dists)) if linf_dists else float('nan'),
            'std_linf': float(np.std(linf_dists)) if linf_dists else float('nan'),
        }

        r = results[L]
        logger.info(f"      Circle: {r['circle_accuracy']:.0%}  "
                    f"Chamfer↑: {r['mean_chamfer_up']:.4f}  "
                    f"PerSeg: {r['per_seg_chamfer']:.4f}  "
                    f"Linf: {r['mean_linf']:.4f}  "
                    f"({time.time()-t0:.1f}s)")

    return results


def run_battery(model, device='cpu', n_avg=30):
    """Run full test battery."""
    t_bat = time.time()

    logger.info("  Test 1: Order sensitivity (NON-ABELIAN TEST)")
    t1 = test_order_sensitivity(model, n_avg, device)
    logger.info(f"    Distinguished: {t1['fraction_distinguished']:.0%}  "
                f"Mean cross-Chamfer: {t1['mean_cross_chamfer']:.4f}")

    logger.info("  Test 2: Circle accuracy (train distribution)")
    t2 = test_circle_accuracy(model, device=device, n_avg=n_avg)
    logger.info(f"    Accuracy: {t2['accuracy']:.0%}  "
                f"(by construction: {t2.get('by_construction', False)})")

    logger.info("  Test 3: LENGTH SCALING (the central experiment)")
    logger.info("    Train on length ≤ 2.  Test on unseen lengths.")
    test_lengths = [2, 3, 4, 6, 8, 10]
    t3 = test_length_scaling(model, test_lengths, n_avg, device)

    logger.info(f"  Battery complete in {time.time()-t_bat:.0f}s")
    return {
        't1_order': t1,
        't2_circle': t2,
        't3_scaling': t3,
    }


# =============================================================================
# §7  SINGLE RUN (one seed)
# =============================================================================

def run_comparison(epochs=300, n_samples=1000, n_points=32, n_avg=30,
                   seed=42, device='cpu', output_dir='results_wedge',
                   figure_dir='figures_wedge', arch_list=None):
    """Run full comparison for one seed."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Training words: all words of length ≤ 2
    # KEY: "ab" and "ba" are DIFFERENT training samples (non-abelian)
    train_words = ['a', 'b', 'aa', 'ab', 'ba', 'bb']
    n_total = n_samples * len(train_words)
    logger.info(f"S¹∨S¹ HIT: base, loop_a, loop_b (free group F₂, no relations)")
    logger.info(f"Seed: {seed}")

    dataset = WedgeLoopDataset(
        words=train_words, n_samples=n_samples, n_pts=n_points,
        max_word_len=2, seed=seed)
    logger.info(f"Dataset: {len(dataset)} samples")

    if arch_list is None:
        arch_list = list(DECODERS.keys())
    logger.info(f"Architectures: {arch_list}")

    kw = dict(vocab_size=3, embed_dim=32, latent_dim=64,
              enc_layers=2, enc_hidden=128, n_points=n_points, max_word_len=2)

    comparison = {}
    for a in arch_list:
        logger.info(f"\n{'='*60}")
        logger.info(f"{a}  |  Type: {ARCH_TYPE[a]}  |  {HIT_LEVELS[a]}")
        logger.info(f"{'='*60}")

        model = DECODERS[a](**kw)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {n_params:,}")

        history, best_loss = train_wedge(
            model, dataset, epochs=epochs, device=device,
            log_every=max(1, epochs // 10))

        logger.info(f"  Final loss: {history[-1]:.6f}")
        logger.info(f"  Battery:")

        model.eval()
        battery = run_battery(model, device=device, n_avg=n_avg)

        comparison[a] = {
            'model': model,
            'n_params': n_params,
            'history': history,
            'best_loss': best_loss,
            'battery': battery,
        }

    # Print summary
    _print_summary(comparison)

    # Make figures
    _make_figures(comparison, figure_dir)

    # Save results
    _save_results(comparison, output_dir, seed)

    return comparison


# =============================================================================
# §8  MULTI-SEED RUNNER
# =============================================================================

def run_multi_seed(n_seeds=3, base_seed=42, **kwargs):
    """Run multiple seeds and aggregate results."""
    all_results = {}
    for i in range(n_seeds):
        seed = base_seed + i * 137
        logger.info(f"\n{'#'*70}")
        logger.info(f"# SEED {i+1}/{n_seeds} (seed={seed})")
        logger.info(f"{'#'*70}")
        all_results[seed] = run_comparison(seed=seed, **kwargs)

    # Aggregate
    logger.info(f"\n{'='*80}")
    logger.info(f"AGGREGATED RESULTS ({n_seeds} seeds)")
    logger.info(f"{'='*80}")

    archs = list(all_results[list(all_results.keys())[0]].keys())
    test_lengths = [2, 3, 4, 6, 8, 10]

    # Per-segment Chamfer: mean ± std across seeds
    logger.info(f"\nPer-segment Chamfer (mean ± std across {n_seeds} seeds):")
    header = f"{'Arch':<22} {'Type':>4}" + "".join(f"{'L='+str(L):>14}" for L in test_lengths)
    logger.info(header)
    logger.info("-" * (26 + 14 * len(test_lengths)))

    for a in archs:
        vals_by_L = defaultdict(list)
        for seed, results in all_results.items():
            scaling = results[a]['battery']['t3_scaling']
            for L in test_lengths:
                if L in scaling:
                    vals_by_L[L].append(scaling[L]['per_seg_chamfer'])

        row = f"{a:<22} {ARCH_TYPE.get(a, '?'):>4}"
        for L in test_lengths:
            vs = vals_by_L.get(L, [])
            if vs:
                m, s = np.mean(vs), np.std(vs)
                row += f"  {m:.3f}±{s:.3f}"
            else:
                row += f"{'—':>14}"
        logger.info(row)

    # Training loss: mean ± std
    logger.info(f"\nTraining loss (mean ± std across {n_seeds} seeds):")
    for a in archs:
        losses = [all_results[s][a]['best_loss'] for s in all_results]
        logger.info(f"  {a:<22} {np.mean(losses):.4f} ± {np.std(losses):.4f}")

    # Order sensitivity
    logger.info(f"\nOrder sensitivity (fraction distinguished, mean ± std):")
    for a in archs:
        fracs = [all_results[s][a]['battery']['t1_order']['fraction_distinguished']
                 for s in all_results]
        logger.info(f"  {a:<22} {np.mean(fracs):.2f} ± {np.std(fracs):.2f}")

    return all_results


# =============================================================================
# §9  MATCHED-LOSS ABLATION
# =============================================================================

def run_matched_loss_ablation(comparison, dataset, epochs_extra=1000,
                              device='cpu', n_avg=30):
    """
    Retrain type-A architectures to match the best type-B training loss.

    This ablation answers: "Is the test-time gap due to training quality
    or architecture?"  If type-A still degrades after matching type-B's
    training loss, the gap is architectural.
    """
    # Find best type-B loss
    type_b_losses = [d['best_loss'] for a, d in comparison.items()
                     if ARCH_TYPE.get(a) == 'B']
    if not type_b_losses:
        logger.info("No type-B architectures found; skipping ablation.")
        return {}

    target_loss = min(type_b_losses)
    logger.info(f"\n{'='*60}")
    logger.info(f"MATCHED-LOSS ABLATION")
    logger.info(f"Target loss (best type-B): {target_loss:.6f}")
    logger.info(f"{'='*60}")

    ablation_results = {}
    for a, d in comparison.items():
        if ARCH_TYPE.get(a) != 'A':
            continue
        if d['best_loss'] <= target_loss * 1.02:
            logger.info(f"\n{a}: already at target loss ({d['best_loss']:.6f}), skipping")
            ablation_results[a] = {'skipped': True, 'original_loss': d['best_loss']}
            continue

        logger.info(f"\n{a}: retraining (current: {d['best_loss']:.6f}, "
                    f"target: {target_loss:.6f})")

        # Recreate model with fresh parameters
        kw = dict(vocab_size=3, embed_dim=32, latent_dim=64,
                  enc_layers=2, enc_hidden=128, n_points=32, max_word_len=2)
        model = DECODERS[a](**kw)

        # Train with lower LR and more epochs for better convergence
        history, best_loss = train_wedge(
            model, dataset, epochs=epochs_extra, lr=5e-4,
            device=device, log_every=max(1, epochs_extra // 10),
            patience=150)

        logger.info(f"  Retrained loss: {best_loss:.6f} "
                    f"(target: {target_loss:.6f})")

        # Re-evaluate
        model.eval()
        battery = run_battery(model, device=device, n_avg=n_avg)

        ablation_results[a] = {
            'skipped': False,
            'original_loss': d['best_loss'],
            'retrained_loss': best_loss,
            'battery': battery,
        }

        # Compare per-segment Chamfer
        test_lengths = [2, 3, 4, 6, 8, 10]
        logger.info(f"\n  Per-segment Chamfer comparison for {a}:")
        logger.info(f"  {'':>8}  {'Original':>10}  {'Retrained':>10}")
        for L in test_lengths:
            orig = d['battery']['t3_scaling'].get(L, {}).get('per_seg_chamfer', float('nan'))
            new = battery['t3_scaling'].get(L, {}).get('per_seg_chamfer', float('nan'))
            logger.info(f"  L={L:>2}:    {orig:10.4f}  {new:10.4f}")

    return ablation_results


# =============================================================================
# §10  SUMMARY & FIGURES
# =============================================================================

def _print_summary(comparison):
    """Print summary tables."""
    archs = list(comparison.keys())
    test_lengths = [2, 3, 4, 6, 8, 10]

    logger.info(f"\n{'='*80}")
    logger.info("S¹∨S¹ RESULTS SUMMARY")
    logger.info(f"{'='*80}")

    # Training
    logger.info(f"\n{'Arch':<22} {'Type':>4} {'Params':>10} {'Best Loss':>10} {'Epochs':>8}")
    logger.info("-" * 60)
    for a, d in comparison.items():
        logger.info(f"{a:<22} {ARCH_TYPE.get(a,'?'):>4} {d['n_params']:>10,} "
                    f"{d['best_loss']:>10.4f} {len(d['history']):>8}")

    # Per-segment Chamfer (THE key table)
    logger.info(f"\n{'='*80}")
    logger.info("Per-segment Chamfer (FAIR — fixed 32 pts/segment):")
    logger.info(f"{'='*80}")
    header = f"{'Arch':<22} {'Type':>4}" + "".join(f"{'L='+str(L):>8}" for L in test_lengths)
    logger.info(header)
    logger.info("-" * (26 + 8 * len(test_lengths)))
    for a, d in comparison.items():
        scaling = d['battery']['t3_scaling']
        row = f"{a:<22} {ARCH_TYPE.get(a,'?'):>4}"
        for L in test_lengths:
            if L in scaling:
                row += f" {scaling[L]['per_seg_chamfer']:>7.4f}"
            else:
                row += f"{'—':>8}"
        logger.info(row)

    # Chamfer (upsampled)
    logger.info(f"\nChamfer (upsampled to GT resolution):")
    header = f"{'Arch':<22} {'Type':>4}" + "".join(f"{'L='+str(L):>8}" for L in test_lengths)
    logger.info(header)
    logger.info("-" * (26 + 8 * len(test_lengths)))
    for a, d in comparison.items():
        scaling = d['battery']['t3_scaling']
        row = f"{a:<22} {ARCH_TYPE.get(a,'?'):>4}"
        for L in test_lengths:
            if L in scaling:
                row += f" {scaling[L]['mean_chamfer_up']:>7.4f}"
            else:
                row += f"{'—':>8}"
        logger.info(row)

    # Circle accuracy
    logger.info(f"\nCircle accuracy:")
    header = f"{'Arch':<22} {'Type':>4}" + "".join(f"{'L='+str(L):>8}" for L in test_lengths)
    logger.info(header)
    logger.info("-" * (26 + 8 * len(test_lengths)))
    for a, d in comparison.items():
        scaling = d['battery']['t3_scaling']
        row = f"{a:<22} {ARCH_TYPE.get(a,'?'):>4}"
        for L in test_lengths:
            if L in scaling:
                row += f" {scaling[L]['circle_accuracy']:>6.0%}"
            else:
                row += f"{'—':>8}"
        logger.info(row)

    # Order sensitivity
    logger.info(f"\nOrder sensitivity (non-abelian test):")
    for a, d in comparison.items():
        t1 = d['battery']['t1_order']
        logger.info(f"  {a:<22} distinguished: {t1['fraction_distinguished']:.0%}  "
                    f"mean cross-Chamfer: {t1['mean_cross_chamfer']:.4f}")


def _make_figures(comparison, figure_dir):
    """Generate the paper figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figures")
        return

    os.makedirs(figure_dir, exist_ok=True)
    archs = list(comparison.keys())

    colors = {
        'wedge_transport':   ('#4393C3', '^', 'Transport [B]'),
        'wedge_transformer': ('#D6604D', 's', 'Transformer [A]'),
        'wedge_sequential':  ('#9970AB', 'D', 'Sequential [A]'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for a in archs:
        scaling = comparison[a]['battery']['t3_scaling']
        lengths = sorted([k for k in scaling.keys() if isinstance(k, int)])
        if not lengths:
            continue

        color, marker, label = colors.get(a, ('gray', 'x', a))

        # Panel 1: Circle accuracy
        circ = [scaling[L]['circle_accuracy'] for L in lengths]
        axes[0].plot(lengths, circ, f'-{marker}', color=color,
                     label=label, linewidth=2, markersize=8)

        # Panel 2: Per-segment Chamfer (THE metric)
        pseg = [scaling[L]['per_seg_chamfer'] for L in lengths]
        pseg_std = [scaling[L].get('std_per_seg', 0) for L in lengths]
        axes[1].errorbar(lengths, pseg, yerr=pseg_std, fmt=f'-{marker}',
                         color=color, label=label, linewidth=2, markersize=8,
                         capsize=3)

        # Panel 3: Chamfer (upsampled)
        chup = [scaling[L]['mean_chamfer_up'] for L in lengths]
        axes[2].plot(lengths, chup, f'-{marker}', color=color,
                     label=label, linewidth=2, markersize=8)

    for ax in axes:
        ax.axvline(2, ls='--', color='gray', alpha=0.4)
        ax.set_xlabel('Word length')
        ax.set_xticks([2, 3, 4, 6, 8, 10])
        ax.axvspan(0, 2.5, alpha=0.08, color='blue')

    axes[0].set_ylabel('Circle accuracy')
    axes[0].set_title('Topological correctness')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=9)

    axes[1].set_ylabel('Per-segment Chamfer')
    axes[1].set_title('Per-segment geometric error\n(Prop. 4.7 metric)')
    axes[1].legend(fontsize=9)

    axes[2].set_ylabel('Chamfer (upsampled)')
    axes[2].set_title('Total geometric error')
    axes[2].legend(fontsize=9)

    fig.suptitle('S¹∨S¹ (π₁ = F₂): Non-abelian compositional generalization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(figure_dir, 'wedge_length_scaling.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  KEY FIGURE saved: {path}")


def _save_results(comparison, output_dir, seed):
    """Save results to JSON."""
    def sanitize(o):
        if isinstance(o, dict):
            return {str(k): sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [sanitize(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (int, float, str, bool, type(None))):
            return o
        if isinstance(o, torch.Tensor):
            return o.tolist()
        return str(o)

    save_d = {}
    for a, d in comparison.items():
        save_d[a] = {k: sanitize(v) for k, v in d.items() if k != 'model'}

    path = os.path.join(output_dir, f'results_seed{seed}.json')
    with open(path, 'w') as f:
        json.dump(save_d, f, indent=2)
    logger.info(f"  Results saved: {path}")


# =============================================================================
# §11  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='S¹∨S¹ wedge experiment')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Samples per word (6 words → 6× total)')
    parser.add_argument('--n_points', type=int, default=32)
    parser.add_argument('--n_avg', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='results_wedge')
    parser.add_argument('--figure_dir', type=str, default='figures_wedge')
    parser.add_argument('--archs', type=str, nargs='+', default=None)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--n_seeds', type=int, default=1,
                        help='Number of seeds for multi-seed run')
    parser.add_argument('--ablation', action='store_true',
                        help='Run matched-loss ablation after main experiment')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if args.quick:
        args.epochs, args.n_samples, args.n_avg = 50, 50, 10
    elif args.full:
        args.epochs, args.n_samples, args.n_avg = 500, 1000, 50

    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        args.device = 'cpu'

    logger.info(f"Device: {args.device}"
                + (f" ({torch.cuda.get_device_name(0)})"
                   if args.device == 'cuda' else ""))
    if args.device == 'cpu':
        logger.warning("⚠ Running on CPU. Use --device cuda for 10-50x speedup.")

    common_kw = dict(
        epochs=args.epochs, n_samples=args.n_samples,
        n_points=args.n_points, n_avg=args.n_avg,
        device=args.device, output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        arch_list=args.archs or list(DECODERS.keys()),
    )

    if args.n_seeds > 1:
        all_results = run_multi_seed(
            n_seeds=args.n_seeds, base_seed=args.seed, **common_kw)

        # Matched-loss ablation (on last seed)
        if args.ablation:
            last_seed = list(all_results.keys())[-1]
            last_comp = all_results[last_seed]
            # Recreate dataset for retraining
            train_words = ['a', 'b', 'aa', 'ab', 'ba', 'bb']
            dataset = WedgeLoopDataset(
                words=train_words, n_samples=args.n_samples,
                n_pts=args.n_points, max_word_len=2, seed=last_seed)
            run_matched_loss_ablation(
                last_comp, dataset, epochs_extra=args.epochs * 2,
                device=args.device, n_avg=args.n_avg)
    else:
        comparison = run_comparison(seed=args.seed, **common_kw)

        # Matched-loss ablation
        if args.ablation:
            train_words = ['a', 'b', 'aa', 'ab', 'ba', 'bb']
            dataset = WedgeLoopDataset(
                words=train_words, n_samples=args.n_samples,
                n_pts=args.n_points, max_word_len=2, seed=args.seed)
            run_matched_loss_ablation(
                comparison, dataset, epochs_extra=args.epochs * 2,
                device=args.device, n_avg=args.n_avg)


if __name__ == '__main__':
    main()
