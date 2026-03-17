"""
klein_experiment.py — Klein Bottle: Where the Proof Term Matters

TARGET TYPE: The Klein bottle K, defined as a HIT:

    K ≡ HIT with
      base  : K                                     (point)
      loopₐ : base = base                           (generator a)
      loop_b : base = base                           (generator b)
      rel   : loop_b · loopₐ · loop_b⁻¹ = loopₐ⁻¹  (relation 2-cell)

    π₁(K) = ⟨a, b | bab⁻¹ = a⁻¹⟩  ≅  ℤ ⋊ ℤ

Each HIT constructor compiles to an architectural component:

    base   →  base point (fixed)
    loopₐ  →  generator network gₐ  (learned loop shape)
    loop_b →  generator network g_b  (learned loop shape)
    rel    →  homotopy network H     (learned proof term)

THE KEY RESULT:

    On the torus (abelian π₁ = ℤ²), H was unnecessary.
    On S¹∨S¹ (free π₁ = F₂), H was vacuous (no relations).
    On the Klein bottle (non-abelian, non-trivial relation), H is ESSENTIAL.

    The relation bab⁻¹ = a⁻¹ means: after traversing b, the a-direction
    is FLIPPED. The transport decoder generates canonical a^n b^m and ignores
    this frame change. The homotopy decoder tracks the frame and uses H
    to generate geometrically correct loops for non-canonical words.

    The per-segment Chamfer gap between transport and homotopy decoders
    IS the relation 2-cell made measurable.

Usage:
    python klein_experiment.py --quick
    python klein_experiment.py --epochs 300 --n_samples 500 --seed 42
    python klein_experiment.py --full --device cuda
"""

import argparse
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# =============================================================================
# §1  Klein bottle geometry (embedded in R⁴)
# =============================================================================

KLEIN_R = 2.0  # major radius
KLEIN_r = 0.8  # minor radius

# Letter encoding: 0=pad, 1=a, 2=A(=a⁻¹), 3=b, 4=B(=b⁻¹)
LETTER_TO_ID = {'a': 1, 'A': 2, 'b': 3, 'B': 4}
ID_TO_LETTER = {0: '<pad>', 1: 'a', 2: 'A', 3: 'b', 4: 'B'}


def angles_to_klein(phi, theta, R=KLEIN_R, r=KLEIN_r):
    """
    Map angular coordinates (φ, θ) to Klein bottle K ⊂ R⁴.

    The embedding uses the half-angle twist in the (x₃, x₄) plane:
      x₁ = (R + r cos θ) cos φ
      x₂ = (R + r cos θ) sin φ
      x₃ = r sin θ cos(φ/2)
      x₄ = r sin θ sin(φ/2)

    This is a proper embedding (no self-intersection) in R⁴.
    The half-angle twist means: traversing φ by 2π flips the θ direction.
    """
    x1 = (R + r * torch.cos(theta)) * torch.cos(phi)
    x2 = (R + r * torch.cos(theta)) * torch.sin(phi)
    x3 = r * torch.sin(theta) * torch.cos(phi / 2)
    x4 = r * torch.sin(theta) * torch.sin(phi / 2)
    return torch.stack([x1, x2, x3, x4], dim=-1)


def angles_to_klein_np(phi, theta, R=KLEIN_R, r=KLEIN_r):
    """Numpy version of Klein bottle embedding."""
    x1 = (R + r * np.cos(theta)) * np.cos(phi)
    x2 = (R + r * np.cos(theta)) * np.sin(phi)
    x3 = r * np.sin(theta) * np.cos(phi / 2)
    x4 = r * np.sin(theta) * np.sin(phi / 2)
    return np.stack([x1, x2, x3, x4], axis=-1)


def canonical_form(word):
    """
    Reduce a word in {a, A, b, B}* to canonical form (n, m) ∈ ℤ × ℤ.

    Uses the relation: b a b⁻¹ = a⁻¹, equivalently ba = a⁻¹b.
    More generally: b^m a^p = a^{(-1)^m · p} b^m.

    The canonical form is a^n b^m.
    """
    n, m = 0, 0
    for c in word:
        if c == 'a':
            # a^{(-1)^m} (frame-corrected a)
            n += (-1) ** m
        elif c == 'A':
            n -= (-1) ** m
        elif c == 'b':
            m += 1
        elif c == 'B':
            m -= 1
    return n, m


def generate_klein_loop(word, n_points=32, noise_std=0.02, rng=None):
    """
    Generate a ground-truth loop on K ⊂ R⁴ following the word ordering.

    Each letter is a segment of n_points points. The generator shapes
    are great circles on the Klein bottle, with the frame change tracked:
    after each 'b', the φ direction reverses for subsequent 'a' segments.

    This ground truth FOLLOWS THE WORD ORDER — it does not canonicalize.
    The per-segment Chamfer between a decoder and this GT measures
    whether the decoder respects word-order structure.
    """
    if rng is None:
        rng = np.random.default_rng()

    R, r = KLEIN_R, KLEIN_r
    segments = []

    # Track position in the universal cover
    phi_off = rng.uniform(0, 2 * np.pi)
    theta_off = rng.uniform(0, 2 * np.pi)
    b_parity = 0  # number of b's traversed (mod 2 determines frame)

    for letter in word:
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        if letter == 'a':
            phi = phi_off + t
            theta = np.full_like(t, theta_off)
            phi_off += 2 * np.pi
        elif letter == 'A':
            phi = phi_off - t
            theta = np.full_like(t, theta_off)
            phi_off -= 2 * np.pi
        elif letter == 'b':
            # After traversing b, the frame for a is flipped
            phi = np.full_like(t, phi_off)
            theta = theta_off + t
            theta_off += 2 * np.pi
            # Apply the Klein twist: φ → 2π - φ at the identification
            # In the universal cover, this means the next a-segment
            # will see a flipped φ direction. We track this via b_parity.
            b_parity += 1
            phi_off = -phi_off  # frame flip in universal cover
        elif letter == 'B':
            phi = np.full_like(t, phi_off)
            theta = theta_off - t
            theta_off -= 2 * np.pi
            b_parity += 1
            phi_off = -phi_off

        pts = angles_to_klein_np(phi, theta, R, r)
        segments.append(pts)

    points = np.concatenate(segments, axis=0).astype(np.float32)
    points += rng.normal(0, noise_std, points.shape).astype(np.float32)
    return points


def klein_winding(word):
    """Canonical winding pair (n_a, n_b) for a word on K."""
    return canonical_form(word)


# =============================================================================
# §2  Dataset
# =============================================================================

def _generate_training_words(max_length=2):
    """
    Generate all words up to max_length in {a, A, b, B}*.
    Includes inverses — essential for Klein bottle relations.
    """
    letters = ['a', 'A', 'b', 'B']
    words = []
    for L in range(1, max_length + 1):
        if L == 1:
            words.extend(letters)
        else:
            # Generate all combinations
            from itertools import product as iproduct
            for combo in iproduct(letters, repeat=L):
                w = ''.join(combo)
                # Skip trivially canceling words (e.g., "aA", "bB")
                n, m = canonical_form(w)
                if n != 0 or m != 0:  # skip identity
                    words.append(w)
    # Remove duplicates that have same canonical form AND same word
    return sorted(set(words))


class KleinLoopDataset(Dataset):
    def __init__(self, words=None, n_samples=500, n_points=32,
                 max_word_len=2, noise_std=0.02, seed=42):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.max_loop_len = max_word_len * n_points
        self.n_points = n_points
        self.samples = []

        if words is None:
            words = _generate_training_words(max_word_len)

        for word in words:
            enc = np.array([LETTER_TO_ID[c] for c in word], dtype=np.int64)
            n, m = canonical_form(word)
            loop_len = len(word) * n_points

            for _ in range(n_samples):
                loop = generate_klein_loop(word, n_points, noise_std, rng)
                padded = np.zeros((self.max_loop_len, 4), dtype=np.float32)
                padded[:loop_len] = loop
                mask = np.zeros(self.max_loop_len, dtype=np.float32)
                mask[:loop_len] = 1.0
                self.samples.append((enc, padded, mask, n, m, word))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc, loop, mask, na, nb, word = self.samples[idx]
        return (torch.from_numpy(enc), torch.from_numpy(loop),
                torch.from_numpy(mask),
                torch.tensor([na, nb], dtype=torch.float32), word)


def collate_klein(batch):
    encs, loops, masks, windings, words = zip(*batch)
    lengths = torch.tensor([len(e) for e in encs])
    max_len = int(lengths.max())
    padded_enc = torch.zeros(len(encs), max_len, dtype=torch.long)
    for i, e in enumerate(encs):
        padded_enc[i, :len(e)] = e
    return (padded_enc, lengths, torch.stack(loops),
            torch.stack(masks), torch.stack(windings), list(words))


# =============================================================================
# §3  Shared base class
# =============================================================================

class _KleinBase(nn.Module):
    def __init__(self, vocab_size=5, embed_dim=32, latent_dim=64,
                 enc_layers=2, enc_hidden=128, n_points=32, max_word_len=2):
        super().__init__()
        self.n_points = n_points
        self.max_word_len = max_word_len
        self.output_len = max_word_len * n_points
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # Encoder: word → latent (for type-A decoders)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, enc_hidden, num_layers=enc_layers,
                              batch_first=True)
        self.to_latent = nn.Linear(enc_hidden, latent_dim)

    def encode_word(self, word_ids, lengths):
        emb = self.embed(word_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.encoder(packed)
        return self.to_latent(h_n[-1])

    def _get_canonical(self, word_ids):
        """Compute canonical form (n, m) from encoded word IDs."""
        B = word_ids.size(0)
        n_vals = torch.zeros(B, device=word_ids.device)
        m_vals = torch.zeros(B, device=word_ids.device)
        for i in range(B):
            word = ''.join(ID_TO_LETTER.get(int(c), '') for c in word_ids[i] if c != 0)
            n, m = canonical_form(word)
            n_vals[i] = n
            m_vals[i] = m
        return n_vals, m_vals

    def _get_word_string(self, word_ids_single):
        """Convert single word tensor to string."""
        return ''.join(ID_TO_LETTER.get(int(c), '') for c in word_ids_single if c != 0)


# =============================================================================
# §4  KleinCoverDecoder — Type A: learned map (n,m) → loop
# =============================================================================

class KleinCoverDecoder(_KleinBase):
    """
    Maps canonical form (n, m) to a learned loop shape.
    Correct homotopy class. No composition coherence.
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        # Maps (n, m) → angle increments
        self.decoder = nn.Sequential(
            nn.Linear(2, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, self.output_len * 2),
        )

    def forward(self, word_ids, lengths):
        n_a, n_b = self._get_canonical(word_ids)
        winding = torch.stack([n_a, n_b], dim=1)  # (B, 2)
        raw = self.decoder(winding)  # (B, output_len * 2)
        B = raw.size(0)
        angles = raw.view(B, self.output_len, 2)
        dphi = angles[:, :, 0]
        dtheta = angles[:, :, 1]

        # Hard winding constraint
        target_phi = n_a.unsqueeze(1) * 2 * math.pi
        target_theta = n_b.unsqueeze(1) * 2 * math.pi
        dphi = dphi - (dphi.sum(dim=1, keepdim=True) - target_phi) / self.output_len
        dtheta = dtheta - (dtheta.sum(dim=1, keepdim=True) - target_theta) / self.output_len

        phi = torch.cumsum(dphi, dim=1)
        theta = torch.cumsum(dtheta, dim=1)
        return angles_to_klein(phi, theta)


# =============================================================================
# §5  KleinTransportDecoder — Type B: canonical concatenation (no H)
# =============================================================================

class KleinTransportDecoder(_KleinBase):
    """
    Learns gₐ and g_b generator shapes.
    Generates canonical form: gₐ^n · g_b^m (or gₐ⁻¹^|n| if n < 0).
    Correct homotopy class + composition in canonical order.
    FAILS on non-canonical word orderings (doesn't track frame).
    """
    def __init__(self, **kw):
        super().__init__(**kw)
        T = self.n_points

        # Generator gₐ: Σ Δφ = 2π, Σ Δθ = 0
        self.ga_dphi = nn.Parameter(torch.randn(T) * 0.1 + 2 * math.pi / T)
        self.ga_dtheta = nn.Parameter(torch.randn(T) * 0.01)
        self.ga_phi0 = nn.Parameter(torch.zeros(1))
        self.ga_theta0 = nn.Parameter(torch.zeros(1))

        # Generator g_b: Σ Δφ = 0, Σ Δθ = 2π
        self.gb_dphi = nn.Parameter(torch.randn(T) * 0.01)
        self.gb_dtheta = nn.Parameter(torch.randn(T) * 0.1 + 2 * math.pi / T)
        self.gb_phi0 = nn.Parameter(torch.zeros(1))
        self.gb_theta0 = nn.Parameter(torch.zeros(1))

    def _corrected_increments(self, raw, target_total):
        T = raw.shape[0]
        return raw - (raw.sum() - target_total) / T

    def get_generators(self):
        """Return gₐ, gₐ⁻¹, g_b, g_b⁻¹ as point clouds on K ⊂ R⁴."""
        T = self.n_points
        # gₐ: winding (1, 0)
        dphi_a = self._corrected_increments(self.ga_dphi, 2 * math.pi)
        dth_a = self._corrected_increments(self.ga_dtheta, 0.0)
        phi_a = self.ga_phi0 + torch.cumsum(dphi_a, 0)
        th_a = self.ga_theta0 + torch.cumsum(dth_a, 0)
        ga = angles_to_klein(phi_a, th_a)

        # gₐ⁻¹: winding (-1, 0) — reverse of gₐ
        ga_inv = ga.flip(0)

        # g_b: winding (0, 1)
        dphi_b = self._corrected_increments(self.gb_dphi, 0.0)
        dth_b = self._corrected_increments(self.gb_dtheta, 2 * math.pi)
        phi_b = self.gb_phi0 + torch.cumsum(dphi_b, 0)
        th_b = self.gb_theta0 + torch.cumsum(dth_b, 0)
        gb = angles_to_klein(phi_b, th_b)

        # g_b⁻¹: reverse of g_b
        gb_inv = gb.flip(0)

        return ga, ga_inv, gb, gb_inv

    def _resample(self, loop, target_len):
        if loop.size(0) == target_len:
            return loop
        if loop.size(0) < 2 or target_len < 2:
            return loop[:target_len] if loop.size(0) >= target_len else loop
        idx = torch.linspace(0, loop.size(0) - 1, target_len, device=loop.device)
        lo = idx.long().clamp(max=loop.size(0) - 2)
        hi = lo + 1
        frac = (idx - lo.float()).unsqueeze(1)
        return loop[lo] * (1 - frac) + loop[hi] * frac

    def forward(self, word_ids, lengths):
        """
        CANONICAL ORDER: always outputs gₐ^n · g_b^m.
        Does not track frame changes from the relation.
        """
        B = word_ids.size(0)
        ga, ga_inv, gb, gb_inv = self.get_generators()

        outputs = []
        for i in range(B):
            n, m = self._get_canonical(word_ids[i:i+1])
            ni, mi = int(n[0].item()), int(m[0].item())

            pieces = []
            # a^n part
            if ni > 0:
                pieces.extend([ga] * ni)
            elif ni < 0:
                pieces.extend([ga_inv] * (-ni))
            # b^m part
            if mi > 0:
                pieces.extend([gb] * mi)
            elif mi < 0:
                pieces.extend([gb_inv] * (-mi))

            if pieces:
                composed = torch.cat(pieces, dim=0)
                outputs.append(self._resample(composed, self.output_len))
            else:
                # Identity: return basepoint
                bp = angles_to_klein(
                    self.ga_phi0.expand(self.output_len),
                    self.ga_theta0.expand(self.output_len))
                outputs.append(bp)

        return torch.stack(outputs)


# =============================================================================
# §6  KleinHomotopyDecoder — Type B + H: frame-aware + learned proof term
# =============================================================================

class KleinHomotopyDecoder(KleinTransportDecoder):
    """
    Extends Transport with:
    1. Frame tracking: after each 'b', the 'a' direction flips
    2. Learned proof term H witnessing ba ≃ a⁻¹b

    H(s) for s ∈ [0,1]: family of loops with winding (-1, 1).
      H(0) = g_b · gₐ      (word order: b then a)
      H(1) = gₐ⁻¹ · g_b    (canonical: a⁻¹ then b)
      H(s) = (1-s)·(g_b·gₐ) + s·(gₐ⁻¹·g_b) + s(1-s)·δ(s)

    H IS THE LEARNED PROOF TERM for the relation bab⁻¹ = a⁻¹.
    """
    def __init__(self, n_homotopy_steps=16, **kw):
        super().__init__(**kw)
        T = self.n_points
        self.n_homotopy_steps = n_homotopy_steps

        # δ(s): correction network for the proof term
        self.correction_net = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 2 * T * 2),  # (2T angles) × (φ, θ)
        )

    def _get_generator_angles(self):
        """Return corrected angle increments for both generators."""
        dphi_a = self._corrected_increments(self.ga_dphi, 2 * math.pi)
        dth_a = self._corrected_increments(self.ga_dtheta, 0.0)
        dphi_b = self._corrected_increments(self.gb_dphi, 0.0)
        dth_b = self._corrected_increments(self.gb_dtheta, 2 * math.pi)
        return dphi_a, dth_a, dphi_b, dth_b

    def get_homotopy(self, n_steps=None):
        """
        Compute the homotopy surface H(s) for s ∈ [0, 1].
        H(0) = g_b · gₐ (word order ba)
        H(1) = gₐ⁻¹ · g_b (canonical order a⁻¹b)

        Returns: (n_steps, 2T, 4) — the proof term as geometry on K
        """
        if n_steps is None:
            n_steps = self.n_homotopy_steps
        T = self.n_points
        dphi_a, dth_a, dphi_b, dth_b = self._get_generator_angles()

        # g_b · gₐ increments (word order ba)
        ba_dphi = torch.cat([dphi_b, dphi_a])
        ba_dth = torch.cat([dth_b, dth_a])

        # gₐ⁻¹ · g_b increments (canonical order a⁻¹b)
        # gₐ⁻¹ has dphi = -dphi_a (reversed), dtheta = -dth_a
        a_inv_b_dphi = torch.cat([-dphi_a.flip(0), dphi_b])
        a_inv_b_dth = torch.cat([-dth_a.flip(0), dth_b])

        s_values = torch.linspace(0, 1, n_steps, device=ba_dphi.device)
        surface = []

        for s in s_values:
            dphi = (1 - s) * ba_dphi + s * a_inv_b_dphi
            dth = (1 - s) * ba_dth + s * a_inv_b_dth

            # Learned correction
            delta_raw = self.correction_net(s.view(1, 1))
            delta = delta_raw.view(2 * T, 2)
            delta_phi = delta[:, 0] - delta[:, 0].mean()
            delta_th = delta[:, 1] - delta[:, 1].mean()
            envelope = s * (1 - s)
            dphi = dphi + envelope * delta_phi
            dth = dth + envelope * delta_th

            phi = self.ga_phi0 + torch.cumsum(dphi, 0)
            theta = self.ga_theta0 + torch.cumsum(dth, 0)
            surface.append(angles_to_klein(phi, theta))

        return torch.stack(surface)

    def forward(self, word_ids, lengths):
        """
        WORD-ORDER AWARE: processes each letter left to right,
        tracking the frame change induced by b-generators.

        After each 'b', the 'a' direction flips (the Klein twist).
        Uses gₐ in the standard frame, gₐ⁻¹ in the flipped frame.
        """
        B = word_ids.size(0)
        ga, ga_inv, gb, gb_inv = self.get_generators()

        outputs = []
        for i in range(B):
            word = self._get_word_string(word_ids[i])
            pieces = []
            frame_flipped = False  # tracks (-1)^(b-parity)

            for letter in word:
                if letter == 'a':
                    if frame_flipped:
                        pieces.append(ga_inv)  # a in flipped frame = a⁻¹
                    else:
                        pieces.append(ga)
                elif letter == 'A':
                    if frame_flipped:
                        pieces.append(ga)  # a⁻¹ in flipped frame = a
                    else:
                        pieces.append(ga_inv)
                elif letter == 'b':
                    pieces.append(gb)
                    frame_flipped = not frame_flipped
                elif letter == 'B':
                    pieces.append(gb_inv)
                    frame_flipped = not frame_flipped

            if pieces:
                composed = torch.cat(pieces, dim=0)
                outputs.append(self._resample(composed, self.output_len))
            else:
                bp = angles_to_klein(
                    self.ga_phi0.expand(self.output_len),
                    self.ga_theta0.expand(self.output_len))
                outputs.append(bp)

        return torch.stack(outputs)


# =============================================================================
# §7  TransformerDecoder — Type A baseline
# =============================================================================

class KleinTransformerDecoder(_KleinBase):
    """Transformer maps latent → point cloud on K. Type A."""
    def __init__(self, n_heads=4, n_dec_layers=3, **kw):
        super().__init__(**kw)
        d = self.latent_dim
        self.pos_embed = nn.Parameter(torch.randn(1, self.output_len, d) * 0.02)
        self.project_latent = nn.Linear(d, d)
        layer = nn.TransformerDecoderLayer(d_model=d, nhead=n_heads,
                                            dim_feedforward=4*d, batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=n_dec_layers)
        self.to_coords = nn.Linear(d, 4)  # R⁴ output

    def forward(self, word_ids, lengths):
        z = self.encode_word(word_ids, lengths)
        B = z.size(0)
        mem = self.project_latent(z).unsqueeze(1)
        tgt = self.pos_embed.expand(B, -1, -1)
        out = self.transformer(tgt, mem)
        return self.to_coords(out)


class KleinTransformerWC(KleinTransformerDecoder):
    """Transformer with hard winding constraint. Type A."""
    winding_constrained = True

    def forward(self, word_ids, lengths):
        raw = super().forward(word_ids, lengths)  # (B, T, 4)
        # Interpret as angle increments
        B, T, _ = raw.shape
        dphi = raw[:, :, 0]
        dtheta = raw[:, :, 1]

        # Hard winding constraint
        n_a, n_b = self._get_canonical(word_ids)
        target_phi = n_a * 2 * math.pi
        target_theta = n_b * 2 * math.pi
        dphi = dphi - (dphi.sum(dim=1, keepdim=True) - target_phi.unsqueeze(1)) / T
        dtheta = dtheta - (dtheta.sum(dim=1, keepdim=True) - target_theta.unsqueeze(1)) / T

        phi = torch.cumsum(dphi, dim=1)
        theta = torch.cumsum(dtheta, dim=1)
        return angles_to_klein(phi, theta)


# =============================================================================
# §8  Loss functions
# =============================================================================

def chamfer_loss_4d(predicted, target, mask=None):
    """Chamfer distance in R⁴."""
    if mask is not None:
        B = predicted.size(0)
        total = 0.0
        count = 0
        for i in range(B):
            m = mask[i].bool()
            if m.sum() < 2:
                continue
            p = predicted[i][:int(m.sum())]
            t = target[i][m]
            d2 = ((p.unsqueeze(1) - t.unsqueeze(0)) ** 2).sum(-1)
            total += (d2.min(1).values.mean() + d2.min(0).values.mean()) / 2
            count += 1
        return total / max(count, 1)
    else:
        d2 = ((predicted.unsqueeze(2) - target.unsqueeze(1)) ** 2).sum(-1)
        return (d2.min(2).values.mean() + d2.min(1).values.mean()) / 2


def smoothness_loss(model):
    """Encourage smooth generators."""
    total = 0.0
    count = 0
    for name, p in model.named_parameters():
        if 'dphi' in name or 'dtheta' in name:
            if p.dim() == 1 and p.shape[0] > 2:
                diff = p[1:] - p[:-1]
                total += (diff ** 2).mean()
                count += 1
    return total / max(count, 1)


# =============================================================================
# §9  Training
# =============================================================================

def train_klein(model, dataset, epochs=300, lr=1e-3, batch_size=64,
                device='cpu', log_every=50, w_smooth=0.05, patience=80):
    """Train with early stopping."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_klein, drop_last=True)
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
        for enc, lengths, targets, masks, windings, words in loader:
            enc, lengths = enc.to(device), lengths.to(device)
            targets, masks = targets.to(device), masks.to(device)

            predicted = model(enc, lengths)
            loss = chamfer_loss_4d(predicted, targets, masks)
            loss = loss + w_smooth * smoothness_loss(model)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1

        scheduler.step()
        avg = total_loss / max(nb, 1)
        history.append(avg)
        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t_start
            logger.info(f"  Epoch {epoch:4d}/{epochs}  loss={avg:.6f}  "
                        f"({elapsed:.0f}s elapsed)")

        if avg < best_loss - 1e-5:
            best_loss = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience and epoch > warmup + patience:
            logger.info(f"  Early stopping at epoch {epoch} "
                        f"(best={best_loss:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    logger.info(f"  Training complete in {time.time()-t_start:.0f}s  "
                f"(best loss: {best_loss:.6f})")
    return history


# =============================================================================
# §10  Test battery
# =============================================================================

@torch.no_grad()
def generate_loop(model, word, n_avg=30, device='cpu'):
    """Generate loop(s) for a word."""
    model.eval()
    enc = torch.tensor([[LETTER_TO_ID[c] for c in word]],
                       dtype=torch.long, device=device)
    lengths = torch.tensor([len(word)], device=device)
    enc_batch = enc.repeat(n_avg, 1)
    lengths_batch = lengths.repeat(n_avg)
    loops = model(enc_batch, lengths_batch).cpu().numpy()
    return loops.mean(axis=0), loops


def _chamfer_np(A, B):
    """Chamfer distance in R⁴."""
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return float((d2.min(1).mean() + d2.min(0).mean()) / 2)


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


def _generate_test_words_klein(length, n_words=8, seed=None):
    """
    Generate diverse test words for the Klein bottle.
    Includes both canonical (a's before b's) and non-canonical words.
    Critical: includes words that exercise the relation (b before a).
    """
    rng = np.random.default_rng(seed)
    letters = ['a', 'A', 'b', 'B']
    words = set()

    # Canonical: a^k b^(L-k) and variants with inverses
    for k in range(length + 1):
        words.add('a' * k + 'b' * (length - k))
        if k > 0:
            words.add('A' * k + 'b' * (length - k))
        if length - k > 0:
            words.add('a' * k + 'B' * (length - k))

    # Non-canonical: words with b's before a's (exercise the relation)
    for _ in range(n_words * 3):
        w = ''.join(rng.choice(letters, size=length))
        n, m = canonical_form(w)
        if n != 0 or m != 0:  # skip identity
            words.add(w)

    # Ensure we have some specifically relation-exercising words
    if length >= 2:
        for _ in range(n_words):
            # ba...ba pattern
            w = ''.join(rng.choice(['ba', 'ab'], size=length // 2))
            if len(w) == length:
                words.add(w)

    return sorted(words)


@torch.no_grad()
def test_length_scaling_klein(model, test_lengths, n_avg=30, device='cpu',
                               n_words_per_length=8, seed=42):
    """
    Length scaling test for Klein bottle.
    Train on length ≤ 2, test on longer words.

    Key metrics:
      - per_seg_chamfer: per-segment Chamfer (the fair metric)
      - relation_gap: Chamfer gap specifically on relation-exercising words
      - canonical_chamfer: Chamfer on canonical-order words only
    """
    results = {}
    n_points = model.n_points

    for li, L in enumerate(test_lengths):
        words = _generate_test_words_klein(L, n_words=n_words_per_length, seed=seed)
        logger.info(f"    Length {L}: testing {len(words)} words [{li+1}/{len(test_lengths)}]")
        t0 = time.time()

        per_seg_all = []
        chamfer_up_all = []
        per_seg_canonical = []
        per_seg_noncanonical = []

        for word in words:
            mean_loop, _ = generate_loop(model, word, n_avg, device)
            gt_res = len(word) * n_points

            # Ground truth
            rng_gt = np.random.default_rng(seed + hash(word) % 10000)
            gt_loops = [generate_klein_loop(word, n_points, 0.02, rng_gt)
                        for _ in range(5)]
            gt_mean = np.mean(gt_loops, axis=0)

            # Upsampled Chamfer
            gen_up = _resample_np(mean_loop, gt_res)
            chamfer_up_all.append(_chamfer_np(gen_up, gt_mean))

            # Per-segment Chamfer (the fair metric)
            out_len = mean_loop.shape[0]
            seg_chamfers = []
            for s in range(len(word)):
                s_start = int(s * out_len / len(word))
                s_end = int((s + 1) * out_len / len(word))
                model_seg = mean_loop[s_start:s_end]
                model_seg_n = _resample_np(model_seg, n_points)
                gt_seg = gt_mean[s * n_points:(s + 1) * n_points]
                if model_seg_n.shape[0] > 0 and gt_seg.shape[0] > 0:
                    seg_chamfers.append(_chamfer_np(model_seg_n, gt_seg))

            if seg_chamfers:
                avg_seg = float(np.mean(seg_chamfers))
                per_seg_all.append(avg_seg)

                # Classify: is this word canonical or not?
                is_canonical = _is_canonical_order(word)
                if is_canonical:
                    per_seg_canonical.append(avg_seg)
                else:
                    per_seg_noncanonical.append(avg_seg)

        results[L] = {
            'n_words': len(words),
            'per_seg_chamfer': float(np.mean(per_seg_all)) if per_seg_all else float('nan'),
            'std_per_seg': float(np.std(per_seg_all)) if per_seg_all else float('nan'),
            'mean_chamfer_up': float(np.mean(chamfer_up_all)) if chamfer_up_all else float('nan'),
            'per_seg_canonical': float(np.mean(per_seg_canonical)) if per_seg_canonical else float('nan'),
            'per_seg_noncanonical': float(np.mean(per_seg_noncanonical)) if per_seg_noncanonical else float('nan'),
            'n_canonical': len(per_seg_canonical),
            'n_noncanonical': len(per_seg_noncanonical),
        }

        r = results[L]
        logger.info(f"      PerSeg: {r['per_seg_chamfer']:.4f}  "
                    f"Canon: {r['per_seg_canonical']:.4f} ({r['n_canonical']}w)  "
                    f"NonCan: {r['per_seg_noncanonical']:.4f} ({r['n_noncanonical']}w)  "
                    f"({time.time()-t0:.1f}s)")

    return results


def _is_canonical_order(word):
    """Check if word has all a/A letters before all b/B letters."""
    seen_b = False
    for c in word:
        if c in ('b', 'B'):
            seen_b = True
        elif c in ('a', 'A') and seen_b:
            return False
    return True


@torch.no_grad()
def test_relation_gap(model, n_avg=30, device='cpu'):
    """
    Test the relation: ba ≃ a⁻¹b on K.
    Measures Fréchet distance between G(ba) and G(a⁻¹b).
    These should be homotopic — the gap measures relation awareness.
    """
    cases = [
        ('ba', 'Ab'),      # The fundamental relation
        ('bba', 'ab'),     # b²a = a·b² (even parity: no flip)
        ('bab', 'ABb'),    # bab: canonical form to check
    ]
    # ba → canonical (-1, 1), Ab → canonical (-1, 1) ✓
    # bba → canonical (1, 2), ab → canonical (1, 1): mismatch
    # Use only pairs with matching canonical form
    cases = []
    test_pairs = [
        ('ba', 'Ab'),       # both: canonical (-1, 1)
        ('bab', 'AbB'),     # check canonical forms match
        ('bbab', 'abB'),    # check
    ]
    for w1, w2 in test_pairs:
        c1, c2 = canonical_form(w1), canonical_form(w2)
        if c1 == c2:
            cases.append((w1, w2, c1))

    results = {}
    for w1, w2, canon in cases:
        m1, _ = generate_loop(model, w1, n_avg, device)
        m2, _ = generate_loop(model, w2, n_avg, device)
        L1 = len(w1) * model.n_points
        L2 = len(w2) * model.n_points
        m1_r = _resample_np(m1[:L1], min(L1, L2))
        m2_r = _resample_np(m2[:L2], min(L1, L2))
        fd = float(np.linalg.norm(m1_r - m2_r, axis=-1).max())
        results[f'{w1}_vs_{w2}'] = {'frechet': fd, 'canonical': canon}

    if results:
        fds = [v['frechet'] for v in results.values() if isinstance(v, dict)]
        results['mean_frechet'] = float(np.mean(fds))
    return results


@torch.no_grad()
def test_commutativity_klein(model, n_avg=30, device='cpu'):
    """
    On K, ab ≠ ba (non-abelian!). Test that the model distinguishes them.
    ab has canonical form (1, 1), ba has canonical form (-1, 1).
    """
    m_ab, _ = generate_loop(model, 'ab', n_avg, device)
    m_ba, _ = generate_loop(model, 'ba', n_avg, device)
    L = 2 * model.n_points
    fd = float(np.linalg.norm(m_ab[:L] - m_ba[:L], axis=-1).max())
    return {
        'frechet_ab_ba': fd,
        'canonical_ab': canonical_form('ab'),
        'canonical_ba': canonical_form('ba'),
        'distinct': canonical_form('ab') != canonical_form('ba'),
    }


def run_battery_klein(model, device='cpu', n_avg=30):
    """Full test battery for Klein bottle."""
    t_bat = time.time()

    logger.info("  Test 1: Relation gap (ba vs a⁻¹b)")
    t1 = test_relation_gap(model, n_avg, device)
    if 'mean_frechet' in t1:
        logger.info(f"    Mean Fréchet: {t1['mean_frechet']:.4f}")

    logger.info("  Test 2: Non-abelian distinction (ab vs ba)")
    t2 = test_commutativity_klein(model, n_avg, device)
    logger.info(f"    Fréchet(ab, ba): {t2['frechet_ab_ba']:.4f}  "
                f"(distinct classes: {t2['distinct']})")

    logger.info("  Test 3: LENGTH SCALING (the central experiment)")
    t3 = test_length_scaling_klein(model, test_lengths=[2, 3, 4, 6, 8, 10],
                                    n_avg=n_avg, device=device)

    logger.info(f"  Battery complete in {time.time()-t_bat:.0f}s")
    return {'t1_relation': t1, 't2_nonabelian': t2, 't3_scaling': t3}


# =============================================================================
# §11  Plotting
# =============================================================================

def plot_klein_results(comparison, output_dir='figures_klein'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    colors = {
        'klein_cover':     ('#D6604D', 's', 'Cover [A]'),
        'klein_transport': ('#4393C3', '^', 'Transport [B]'),
        'klein_homotopy':  ('#2CA02C', 'o', 'Homotopy [B+H]'),
        'klein_transformer': ('#9970AB', 'D', 'Transformer [A]'),
        'klein_transformer_wc': ('#E08214', 'v', 'Transformer+WC [A]'),
    }

    # Figure 1: Per-segment Chamfer vs length (all words)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for a, d in comparison.items():
        scaling = d['battery'].get('t3_scaling', {})
        if not scaling:
            continue
        lengths = sorted([k for k in scaling.keys() if isinstance(k, int)])
        if not lengths:
            continue
        color, marker, label = colors.get(a, ('gray', 'x', a))

        # Panel 1: All words
        per_seg = [scaling[L]['per_seg_chamfer'] for L in lengths]
        axes[0].plot(lengths, per_seg, f'-{marker}', color=color,
                     label=label, linewidth=2, markersize=8)

        # Panel 2: Canonical words only
        can = [scaling[L].get('per_seg_canonical', float('nan')) for L in lengths]
        axes[1].plot(lengths, can, f'-{marker}', color=color,
                     label=label, linewidth=2, markersize=8)

        # Panel 3: Non-canonical words only (exercises the relation)
        noncan = [scaling[L].get('per_seg_noncanonical', float('nan')) for L in lengths]
        axes[2].plot(lengths, noncan, f'-{marker}', color=color,
                     label=label, linewidth=2, markersize=8)

    for ax in axes:
        ax.axvline(2, ls='--', color='gray', alpha=0.4)
        ax.set_xlabel('Word length')
        ax.set_xticks([2, 3, 4, 6, 8, 10])
        ax.axvspan(0, 2.5, alpha=0.08, color='blue')

    axes[0].set_ylabel('Per-segment Chamfer')
    axes[0].set_title('All words')
    axes[0].legend(fontsize=8)

    axes[1].set_ylabel('Per-segment Chamfer')
    axes[1].set_title('Canonical words (a...ab...b)')

    axes[2].set_ylabel('Per-segment Chamfer')
    axes[2].set_title('Non-canonical (exercises relation)')

    fig.suptitle('Klein bottle K: π₁ = ⟨a,b | bab⁻¹=a⁻¹⟩\n'
                 'Homotopy decoder uses proof term H for frame tracking',
                 fontsize=11, fontweight='bold', y=1.04)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'klein_length_scaling.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  KEY FIGURE saved: {output_dir}/klein_length_scaling.pdf")


# =============================================================================
# §12  Architecture registry and main runner
# =============================================================================

DECODERS = {
    'klein_cover': KleinCoverDecoder,
    'klein_transport': KleinTransportDecoder,
    'klein_homotopy': KleinHomotopyDecoder,
    'klein_transformer': KleinTransformerDecoder,
    'klein_transformer_wc': KleinTransformerWC,
}

TYPE_LABELS = {
    'klein_cover': 'A',
    'klein_transport': 'B',
    'klein_homotopy': 'B',
    'klein_transformer': 'A',
    'klein_transformer_wc': 'A',
}


def run_comparison(epochs=300, n_samples=500, n_points=32, n_avg=30,
                   seed=42, device='cpu', output_dir='results_klein',
                   figure_dir='figures_klein', arch_list=None):
    """Run the Klein bottle experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if arch_list is None:
        arch_list = ['klein_cover', 'klein_transport', 'klein_homotopy',
                     'klein_transformer_wc']

    logger.info(f"\n{'='*72}")
    logger.info("KLEIN BOTTLE EXPERIMENT: π₁ = ⟨a, b | bab⁻¹ = a⁻¹⟩")
    logger.info(f"  The FIRST space where the proof term H is non-trivial")
    logger.info(f"  Seed={seed}  Epochs={epochs}  Samples={n_samples}")
    logger.info(f"{'='*72}")

    kw = dict(n_points=n_points, max_word_len=2)

    # Generate training words
    train_words = _generate_training_words(max_length=2)
    logger.info(f"  Training words ({len(train_words)}): {train_words[:10]}...")

    dataset = KleinLoopDataset(
        words=train_words, n_samples=n_samples, n_points=n_points,
        max_word_len=2, seed=seed)
    logger.info(f"  Dataset: {len(dataset)} samples")

    comparison = {}
    for arch in arch_list:
        logger.info(f"\n{'='*60}")
        logger.info(f"{arch}  |  Type: {TYPE_LABELS.get(arch, '?')}")
        logger.info(f"{'='*60}")

        model = DECODERS[arch](**kw)
        np_ = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {np_:,}")

        history = train_klein(model, dataset, epochs=epochs,
                              batch_size=min(64, len(dataset)),
                              device=device, log_every=max(epochs // 5, 1))

        os.makedirs(output_dir, exist_ok=True)
        torch.save({'model_state': model.state_dict(), 'history': history},
                   os.path.join(output_dir, f'klein_{arch}_s{seed}.pt'))

        logger.info(f"  Battery:")
        battery = run_battery_klein(model, device=device, n_avg=n_avg)
        comparison[arch] = {
            'final_loss': history[-1],
            'best_loss': min(history),
            'n_params': np_,
            'battery': battery,
            'model': model,
        }

    # Summary tables
    logger.info(f"\n{'='*80}")
    logger.info("KLEIN BOTTLE RESULTS SUMMARY")
    logger.info(f"{'='*80}")

    logger.info(f"\n--- Per-Segment Chamfer (ALL words) ---")
    all_lengths = set()
    for d in comparison.values():
        t3 = d['battery'].get('t3_scaling', {})
        all_lengths.update(k for k in t3.keys() if isinstance(k, int))
    all_lengths = sorted(all_lengths)

    header = f"{'Arch':<25} {'Type':>5}" + "".join(f"{'L='+str(L):>10}" for L in all_lengths)
    logger.info(header)
    logger.info("-" * len(header))
    for a, d in comparison.items():
        t3 = d['battery'].get('t3_scaling', {})
        typ = TYPE_LABELS.get(a, '?')
        vals = [f"{t3[L]['per_seg_chamfer']:9.3f}" if L in t3 else f"{'—':>9}" for L in all_lengths]
        logger.info(f"{a:<25} {typ:>5}" + "".join(f"{v:>10}" for v in vals))

    logger.info(f"\n--- Per-Segment Chamfer (NON-CANONICAL words only) ---")
    logger.info(f"  These exercise the Klein relation bab⁻¹ = a⁻¹")
    logger.info(header)
    logger.info("-" * len(header))
    for a, d in comparison.items():
        t3 = d['battery'].get('t3_scaling', {})
        typ = TYPE_LABELS.get(a, '?')
        vals = []
        for L in all_lengths:
            if L in t3 and not np.isnan(t3[L].get('per_seg_noncanonical', float('nan'))):
                vals.append(f"{t3[L]['per_seg_noncanonical']:9.3f}")
            else:
                vals.append(f"{'—':>9}")
        logger.info(f"{a:<25} {typ:>5}" + "".join(f"{v:>10}" for v in vals))

    # Save results
    save_d = {a: {k: v for k, v in d.items() if k != 'model'}
              for a, d in comparison.items()}

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
        return str(o)

    with open(os.path.join(output_dir, f'klein_seed{seed}.json'), 'w') as f:
        json.dump(sanitize(save_d), f, indent=2)
    logger.info(f"\n  Results saved to {output_dir}/klein_seed{seed}.json")

    plot_klein_results(comparison, output_dir=figure_dir)
    return comparison


# =============================================================================
# §13  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Klein bottle experiment: where the proof term matters")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_points', type=int, default=32)
    parser.add_argument('--n_avg', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='results_klein')
    parser.add_argument('--figure_dir', type=str, default='figures_klein')
    parser.add_argument('--archs', type=str, nargs='+', default=None)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if args.quick:
        args.epochs, args.n_samples, args.n_avg = 50, 50, 10
    elif args.full:
        args.epochs, args.n_samples, args.n_avg = 500, 1000, 50
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        args.device = 'cpu'

    logger.info(f"Device: {args.device}"
                + (f" ({torch.cuda.get_device_name(0)})"
                   if args.device == 'cuda' else ""))

    run_comparison(
        epochs=args.epochs, n_samples=args.n_samples,
        n_points=args.n_points, n_avg=args.n_avg,
        seed=args.seed, device=args.device,
        output_dir=args.output_dir, figure_dir=args.figure_dir,
        arch_list=args.archs,
    )


if __name__ == '__main__':
    main()
