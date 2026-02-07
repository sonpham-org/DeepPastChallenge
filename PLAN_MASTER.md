# Deep Past Challenge - Master Improvement Plan

## Current State
- **Model**: ByT5-small (300M params), two-stage fine-tuned
- **Scores**: Stage1=19.26, Stage2=14.05 (val), **Leaderboard=23.1** (baseline submission)
- **Leaderboard top**: ~38.7 (we need ~15.6 point improvement)
- **Budget**: 5 submissions/day, 9hr inference runtime, ~45 days left

## Leaderboard Intel (Top Competitors Are Using)
1. **Weight-space model averaging** (2-3 models averaged in weight space) — THE #1 differentiator
2. **Aggressive post-processing**: OA Lexicon name normalization, translation memory exact match, repetition removal
3. **Tuned generation**: num_beams=8-10, length_penalty=1.08-1.5, repetition_penalty=1.2
4. **Sentence alignment** of document-level train data to match sentence-level test format
5. **Bidirectional training**: train both Akk→Eng AND Eng→Akk to double data
6. **LLM post-processing** was tried but TURNED OFF by top teams (hurts BLEU via paraphrasing)

---

## PHASE 1: Quick Inference Wins (Days 1-3, No Retraining)

### 1A. Beam Search Tuning [Submission 1-2]
```python
num_beams = 8                # up from 4
length_penalty = 1.1         # tune in [0.8, 1.0, 1.1, 1.2, 1.5]
no_repeat_ngram_size = 3     # CRITICAL: fixes repetition problem
repetition_penalty = 1.2     # keep
early_stopping = True
```

### 1B. Post-Processing Pipeline [Submission 3]
- **Repetition removal**: detect and collapse repeated phrases/n-grams
- **Translation memory**: if test transliteration exactly matches training, use known translation
- **OA Lexicon name normalization**: cross-reference predicted names against OA_Lexicon_eBL.csv
- **Whitespace/Unicode normalization**: NFC normalize (matters for chrF character-level scoring)
- **Fraction normalization**: .5 → 1/2, .33333 → 1/3, etc.

### 1C. MBR Decoding [Submissions 4-5]
```python
# Generate N candidates via epsilon sampling
epsilon_cutoff = 0.02        # best sampling strategy per research
temperature = 1.0
num_return_sequences = 16-32

# Score each candidate against all others using chrF++
# Pick candidate with highest average pairwise chrF++ score
# Libraries: ZurichNLP/mbr or manual with sacrebleu
```
- Use **fastChrF** (Rust-optimized) for speed
- Custom utility matching `sqrt(BLEU * chrF++)` as alternative
- Expected: +2-4 BLEU over beam search

---

## PHASE 2: Training Improvements (Days 3-10)

### 2A. Sentence Alignment [HIGHEST PRIORITY]
- Split document-level training data into sentence-level pairs
- Use `Sentences_Oare_FirstWord_LinNum.csv` as alignment guide
- Match train format to test format (sentence-level)
- Expected: +3-5 BLEU (eliminates train/test distribution mismatch)

### 2B. Scale Up: ByT5-base (580M)
- Top competitors use ByT5-base, not ByT5-small
- 2x parameters = significantly better quality
- Still fits on Kaggle P100 (16GB) with gradient accumulation

### 2C. Bidirectional Training
```python
# Forward: Akkadian -> English
df_fwd['input'] = "translate Akkadian to English: " + src
# Backward: English -> Akkadian
df_bwd['input'] = "translate English to Akkadian: " + tgt
# Doubles training data effectively
```

### 2D. R-Drop Regularization
- Add KL-divergence loss between two forward passes with different dropout masks
- Coefficient alpha = 1.0-5.0
- Expected: +1-3 BLEU (especially impactful for low-resource)
- Zero architecture changes needed

### 2E. Training Hyperparameter Improvements
```python
optimizer = "adafactor"       # memory-efficient, standard for T5
learning_rate = 1e-4          # validated by TACL ByT5 study
label_smoothing = 0.1         # standard, proven +BLEU
weight_decay = 0.05           # stronger regularization for small data
warmup_steps = 4000           # gradual warmup
fp16 = False                  # FP32 required for ByT5 stability
dropout = 0.2                 # slightly higher for small datasets
```

### 2F. Save Multiple Checkpoints for Averaging
- Save every N steps during stable training phase
- Average last 5-10 checkpoints → +0.5-1.0 BLEU (free at inference)

---

## PHASE 3: Data Augmentation (Days 5-15)

### 3A. Back-Translation
1. Train reverse model (English → Akkadian) on existing parallel data
2. Translate English monolingual text (domain-relevant: ancient trade, contracts) into synthetic Akkadian
3. Tag synthetic pairs with `<BT>` prefix
4. Add to training data
- Expected: +2-5 BLEU

### 3B. Glossary-Augmented Prompts
```python
# Prepend relevant lexicon entries to source
input = "[LEX: damqu=good; awilu=man] translate Akkadian to English: {lu₂}a-wi-lum dam-qum..."
```
- Use OA_Lexicon_eBL.csv (39,332 entries)
- Train model to use glossary hints

### 3C. Noise Injection (Simulating Damaged Tablets)
- Randomly replace tokens with `<gap>` (10-20% of examples)
- Random character deletion/insertion
- Improves robustness to real tablet damage

### 3D. LLM-Generated Synthetic Data
- Use Claude/GPT to generate Old Assyrian business letter translations
- Provide lexicon + grammar rules as context
- Even noisy synthetic data helps when resources are low
- Expected: +2-4 chrF

### 3E. Curriculum Learning
- Stage 1a: Formulaic/repetitive texts (contracts, receipts) — predictable patterns
- Stage 1b: General Akkadian (all external data)
- Stage 2: Old Assyrian specialization (competition data)

---

## PHASE 4: Multi-Model Ensemble (Days 10-20)

### 4A. Train Diverse Models for Model Soup
- Train 3-5 ByT5 variants with different hyperparameters:
  - Learning rates: [1e-4, 3e-4, 5e-4]
  - Dropout: [0.1, 0.2, 0.3]
  - With/without R-Drop
  - Different random seeds
- **Weight-space average** (model soup) the best checkpoints
- Zero inference cost, +1-3 BLEU

### 4B. Architecture Diversity
- Train NLLB-200-distilled-600M as ensemble member (very different architecture)
- Use ByT5 + NLLB candidates in MBR reranking pool
- Different architectures = diverse errors = better ensemble

### 4C. Greedy Soup Selection
```python
# Only include models that improve validation score
soup = [best_model]
for model in sorted_models:
    candidate_soup = average(soup + [model])
    if validate(candidate_soup) > validate(soup):
        soup.append(model)
```

---

## PHASE 5: Advanced Inference (Days 15-30)

### 5A. MBR with Custom Utility
```python
def competition_utility(hypothesis, reference):
    bleu = sacrebleu.BLEU().sentence_score(hypothesis, [reference]).score
    chrf = sacrebleu.CHRF(word_order=2).sentence_score(hypothesis, [reference]).score
    return math.sqrt(max(bleu, 0.01) * max(chrf, 0.01))
```

### 5B. Diverse Beam Search + MBR
- Use diverse beam search to generate initial candidates (more diverse than standard beam)
- Then apply MBR reranking on the diverse candidates

### 5C. Multi-Model Candidate Pool
- Generate candidates from ByT5-base soup, ByT5-small soup, NLLB-200
- MBR rerank across all candidates per sentence
- Pick best from the combined pool

### 5D. MBR Self-Distillation (If Time Permits)
1. Generate 50 candidates per training sentence via sampling
2. Select best via MBR with chrF++ utility
3. Fine-tune model on these MBR-selected translations
4. Model learns to produce MBR-quality outputs with standard beam search

---

## Estimated Score Progression

| Phase | Technique | Expected Score |
|-------|-----------|---------------|
| Baseline | Current model (ByT5-small) | **23.1** |
| Phase 1 | Beam tuning + post-processing | ~25-26 |
| Phase 1C | + MBR decoding | ~27-29 |
| Phase 2A | + Sentence alignment | ~29-31 |
| Phase 2B | + ByT5-base (580M) | ~32-34 |
| Phase 2C-E | + Bidirectional + R-Drop + tuning | ~33-35 |
| Phase 3 | + Data augmentation | ~34-36 |
| Phase 4 | + Model soup ensemble | ~36-38 |
| Phase 5 | + Advanced inference | ~37-39+ |

---

## Key References
- [ByT5 vs mT5 for MT (TACL 2024)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00651/120650)
- [Akkadian NMT (PNAS Nexus 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10153418/)
- [R-Drop (NeurIPS 2021)](https://arxiv.org/abs/2106.14448)
- [Tagged Back-Translation](https://arxiv.org/abs/1906.06442)
- [Model Soups (ICML 2022)](https://arxiv.org/abs/2203.05482)
- [MBR Decoding (TACL 2022)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00491/112497)
- [Epsilon Sampling (EMNLP 2023)](https://aclanthology.org/2023.findings-emnlp.617/)
- [ZurichNLP/mbr Library](https://github.com/ZurichNLP/mbr)
- [fastChrF](https://github.com/jvamvas/fastChrF)
- [DPC Starter Train Notebook](https://www.kaggle.com/code/takamichitoda/dpc-starter-train)
- [DPC Weight Averaging Notebook](https://www.kaggle.com/code/hanifnoerrofiq/dpc-weight-averaging-clean-repetition)
- [Test-Time Scaling for MT (2025)](https://arxiv.org/abs/2509.19020)
