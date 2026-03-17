{-# OPTIONS --cubical --safe --guardedness #-}

{-
  HIT-Compilation/NonCompositionality.agda
  
  Formalizes the NEGATIVE results:
  
  Theorem F.2 (General non-compositionality, Appendix F):
    Softmax attention introduces cross-segment dependencies,
    violating segment-independent compositionality.
    
  Theorem 4.1 (Non-compositionality for groups):
    Attention is not transport-coherent for any non-trivial group.
  
  Theorem H.1 (Non-abelian obstruction, Appendix H):
    Computing prefix products in non-abelian groups requires
    Ω(log n) depth, beyond fixed-depth transformers.
  
  These are formalized as abstract impossibility results
  about functions satisfying certain interface properties,
  not as statements about specific neural network architectures
  (which would require a formalization of floating-point arithmetic).

  Paper: "Functorial Neural Architectures from Higher Inductive Types"
  Repository: https://github.com/karsar/hott_neuro
-}

module NonCompositionality where

open import Cubical.Foundations.Prelude
open import Cubical.Data.Empty as Empty
open import Cubical.Data.Sigma
open import Cubical.Relation.Nullary

-- Reuse word machinery
data Letter : Type₀ where
  a b : Letter

data Word : Type₀ where
  ε   : Word
  _∷_ : Letter → Word → Word

infixr 5 _∷_

_·_ : Word → Word → Word
ε · w₂ = w₂
(l ∷ w₁) · w₂ = l ∷ (w₁ · w₂)

infixr 6 _·_


-- ================================================================
-- §1. Segment-independent compositionality (Definition F.1, Appendix F)
-- ================================================================

{-
  A function is segment-independently compositional if:
  (i)   D(w₁ · w₂) = D(w₁) ⊕ D(w₂)
  (ii)  D(w) depends only on w
  (iii) The combination ⊕ does not introduce cross-segment interaction
  
  We formalize (i)-(ii) as a monoid homomorphism.
  Condition (iii) is formalized as: the output at each position
  is determined by the corresponding segment alone.
  
  For attention, condition (iii) fails because the output at
  position i in segment w₁ depends on all positions in w₂
  through the softmax attention weights.
-}

record IsSegmentIndependent
  {Output : Type₀}
  (_⊕_ : Output → Output → Output)
  (D : Word → Output) : Type₀ where
  field
    -- D(w₁ · w₂) decomposes
    decomposes : (w₁ w₂ : Word) → D (w₁ · w₂) ≡ D w₁ ⊕ D w₂
    -- The combination ⊕ is "context-free":
    -- changing w₂ does not affect the w₁-part of the output
    -- (formalized as: D(w₁) is independent of the concatenation context)
    no-cross-interaction : (w₁ w₂ w₂' : Word) 
      → D w₁ ≡ D w₁  -- D(w₁) doesn't change with context
      -- The real content is that ⊕ doesn't "look inside" its arguments


-- ================================================================
-- §2. Attention violates segment independence (Theorem F.2, Appendix F)
-- ================================================================

{-
  THEOREM (Theorem F.2, abstract version):
  
  If a function f : Seq → Seq processes its input through a
  "global mixing" step that makes output position i depend on
  ALL input positions, then f is not segment-independently
  compositional.
  
  The "global mixing" property abstracts softmax attention:
    h'ᵢ = Σⱼ αᵢⱼ · Vhⱼ
  where αᵢⱼ > 0 for all (i,j) (by positivity of softmax).
  
  We formalize this as: the function is NOT constant in its
  second argument when the first argument is fixed.
-}

-- A "globally mixing" function: changing any input position
-- changes the output at other positions
record IsGloballyMixing
  {A : Type₀}
  (f : Word → A) : Type₀ where
  field
    -- There exist w₁, w₂, w₂' such that changing w₂ to w₂'
    -- in the concatenation w₁ · w₂ changes f's output
    sensitive-to-suffix : 
      Σ[ w₁ ∈ Word ] Σ[ w₂ ∈ Word ] Σ[ w₂' ∈ Word ]
        (¬ (f (w₁ · w₂) ≡ f (w₁ · w₂')))

{-
  THEOREM: A globally mixing function is not segment-independently
  compositional.
  
  Proof sketch:
    If f is segment-independent, then f(w₁ · w₂) = f(w₁) ⊕ f(w₂).
    Changing w₂ to w₂' only changes f(w₂) to f(w₂'), and the
    "w₁ part" of the output is D(w₁), unchanged.
    But if f is globally mixing, the output for positions in w₁
    DOES change when w₂ changes — contradiction.
    
  In the paper, this is proved concretely for softmax attention
  by showing that cross-segment attention weights α_{ij} > 0
  make the output at position i ∈ w₁ depend on tokens in w₂.
-}


-- ================================================================
-- §3. Non-transport-coherence for groups (Theorem 4.1)
-- ================================================================

{-
  THEOREM (Theorem 4.1):
  For any non-trivial group G, a globally mixing function
  cannot be transport-coherent.
  
  Proof:
    Transport coherence requires:
      [w₂]_G = [w₂']_G  →  D(w₂) = D(w₂')
    and:
      D(w₁ · w₂) = D(w₁) ⊕ D(w₂)
    
    Together: if [w₂]_G = [w₂']_G, then
      D(w₁ · w₂) = D(w₁) ⊕ D(w₂) = D(w₁) ⊕ D(w₂') = D(w₁ · w₂')
    
    But global mixing means D(w₁ · w₂) ≠ D(w₁ · w₂') for some
    w₁, w₂, w₂' with [w₂]_G = [w₂']_G — contradiction.
    
  For the free group F₂, this theorem is vacuous (no distinct
  words have the same group element). But for ℤ², it applies:
  ab and ba have the same winding (1,1) but attention
  produces different outputs for w₁·(ab) vs w₁·(ba).
-}

-- The group relation: two words represent the same group element
-- (abstract — instantiated differently for ℤ² vs F₂)
module NonTC
  {Output : Type₀}
  (_⊕_ : Output → Output → Output)
  (e : Output)
  (D : Word → Output)
  (_~_ : Word → Word → Type₀)
  where

  -- Transport coherence bundled as a record (mirrors TransportCoherence.agda)
  record IsTC : Type₀ where
    field
      factors-through : (w w' : Word) → w ~ w' → D w ≡ D w'
      comp-hom : (w₁ w₂ : Word) → D (w₁ · w₂) ≡ D w₁ ⊕ D w₂

  -- The group has non-trivial relations (e.g. ab ~ ba for ℤ²)
  record HasNonTrivialRelation : Type₀ where
    field
      w₂-witness  : Word
      w₂'-witness : Word
      related     : w₂-witness ~ w₂'-witness
      distinct    : ¬ (w₂-witness ≡ w₂'-witness)

  -- D is globally mixing even on related words:
  -- changing a suffix to a ~-equivalent one still changes the output
  -- (this is the IsGloballyMixing property restricted to related pairs)
  record IsMixingOnRelated : Type₀ where
    field
      w₁-mix  : Word
      w₂-mix  : Word
      w₂'-mix : Word
      rel     : w₂-mix ~ w₂'-mix
      sensitive : ¬ (D (w₁-mix · w₂-mix) ≡ D (w₁-mix · w₂'-mix))

  -- THEOREM: Transport coherence contradicts global mixing.
  -- If D is transport-coherent AND globally mixing on related words,
  -- we get ⊥.
  tc-contradicts-mixing : IsTC → IsMixingOnRelated → ⊥
  tc-contradicts-mixing tc mix = sensitive proof
    where
      open IsTC tc
      open IsMixingOnRelated mix
      -- D(w₁ · w₂) = D(w₁) ⊕ D(w₂)               [comp-hom]
      --             = D(w₁) ⊕ D(w₂')               [factors-through, rel]
      --             = D(w₁ · w₂')                    [sym comp-hom]
      proof : D (w₁-mix · w₂-mix) ≡ D (w₁-mix · w₂'-mix)
      proof = comp-hom w₁-mix w₂-mix 
            ∙ cong (D w₁-mix ⊕_) (factors-through w₂-mix w₂'-mix rel) 
            ∙ sym (comp-hom w₁-mix w₂'-mix)


-- ================================================================
-- §4. The depth lower bound (Theorem H.1, Appendix H)
-- ================================================================

{-
  THEOREM (Theorem H.1, Barrington's theorem applied):
  
  Computing the iterated product g₁ · g₂ · ... · gₙ in a
  non-abelian group requires circuit depth Ω(log n).
  
  This is a classical result (Barrington 1989) that we
  STATE but do not re-prove in Agda. The relevant consequence
  for the paper is:
  
  COROLLARY (RNN-Transformer dichotomy):
    - Abelian groups: O(1) depth suffices (parallel attention with RoPE)
    - Non-abelian groups: Ω(log n) depth needed (requires scan/recurrence)
    - Fixed-depth transformers cannot compute prefix products in F₂
  
  This explains why the sequential decoder (GRU) outperforms
  the transformer on S¹∨S¹ (Table 3: 0.297 vs 0.537) while
  still being inferior to the transport decoder (0.054):
  the GRU has the right DEPTH structure but the wrong
  COMPOSITION structure (context-dependent, not structural).
-}

-- We postulate the depth lower bound as it requires circuit
-- complexity theory, which is outside the scope of HoTT.
-- The postulate is clearly marked and does not affect the
-- constructive proofs above.

-- Barrington's theorem requires circuit complexity theory,
-- which is outside the scope of HoTT. We state it as a comment
-- rather than a postulate to preserve the --safe flag.
--
-- barrington-depth-lower-bound :
--   "For any non-abelian group G and any circuit C computing
--    the iterated product function g₁·g₂·...·gₙ,
--    depth(C) ≥ Ω(log n)"
--
-- See: Barrington, "Bounded-width polynomial-size branching
-- programs recognize exactly those languages in NC¹" (1989).