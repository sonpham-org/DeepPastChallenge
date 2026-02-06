# Deep Past Challenge - Implementation Plan

## Chosen Approach: Path A - ByT5 Two-Stage Training

### Why ByT5?
- Best proven model for low-resource cuneiform translation
- Byte-level processing = no tokenizer issues with Akkadian characters
- Validated in academic papers (ALP/NAACL 2025, +9.85 chrF++ over mT5 at 400 pairs)
- Already used by multiple top competitors in this challenge

### Architecture
- **Model**: `google/byt5-base` (580M params)
- **Two-stage fine-tuning**:
  1. **Stage 1 - General Akkadian**: Train on all available Akkadian data
     - Akkademia corpus (~56K sentence pairs from ORACC)
     - ORACC Akkadian-English Parallel Corpus (filter for relevant periods)
     - Competition train.csv (1,561 document-level pairs)
     - Lexicon-augmented examples from `OA_Lexicon_eBL.csv`
  2. **Stage 2 - Old Assyrian Specialization**: Fine-tune on competition data only
     - Lower learning rate (5e-5 vs 2e-4)
     - Focus on Old Assyrian business record style

### Data Pipeline
1. Download competition data via Kaggle API
2. Download external datasets:
   - Akkademia corpus from GitHub: https://github.com/gaigutherz/Akkademia
   - ORACC parallel corpus from Kaggle: manwithacat/oracc-akkadian-english-parallel-corpus
3. Preprocess transliterations (remove editorial marks, normalize diacritics, handle gaps)
4. Split documents into sentences for training (align with test format)

### Preprocessing
```
Remove: !, ?, /, :, ., ˹, ˺
Keep content in: < >, << >>
Remove brackets, keep content: [ ]
Replace: [x] → <gap>, ... → <big_gap>
Normalize subscript digits to Unicode
Keep determinatives in curly brackets: {d}, {ki}, {lu₂}
```

### Training Configuration
```python
MODEL_NAME = "google/byt5-base"  # or byt5-small for faster iteration
PREFIX = "translate Akkadian to English: "
MAX_SOURCE_LEN = 1024
MAX_TARGET_LEN = 1024

# Stage 1
STAGE1_EPOCHS = 10
STAGE1_LR = 2e-4
BATCH_SIZE = 20
GRAD_ACC = 2

# Stage 2
STAGE2_EPOCHS = 5
STAGE2_LR = 5e-5
PATIENCE = 3

# Inference
BEAM_WIDTH = 4
REP_PENALTY = 1.2
```

### Inference & Submission
- Beam search with width 4, repetition penalty 1.2
- Generate `submission.csv` with columns: `id`, `translation`
- Must run within 9 hours, no internet access

### Key Kaggle Notebooks to Reference
- [ByT5 Two-Stage Training](https://www.kaggle.com/code/xbar19/deep-past-challenge-byt5-base-training)
- [ByT5 Inference](https://www.kaggle.com/code/xbar19/deep-past-challenge-byt5-base-inference)
- [ByT5 Akkadian Combined v1.0.6](https://www.kaggle.com/code/manwithacat/byt5-akkadian-combined-v1-0-6)
- [Akkadian ByT5 v2 Ensemble](https://www.kaggle.com/code/manwithacat/akkadian-byt5-v2-ensemble)

### Key GitHub Repos
- https://github.com/haakoan/Kaggle-Deep-Past-Challenge-Translate-Akkadian-to-English
- https://github.com/kbsooo/Akkadian-to-English
- https://github.com/gaigutherz/Akkademia

### Potential Enhancements (after baseline)
1. **Back-translation**: Train English→Akkadian model to create synthetic pairs
2. **Glossary-augmented prompts**: Inject relevant dictionary entries into the prompt
3. **Ensemble**: Combine ByT5 with NLLB-200 or MarianMT outputs
4. **Noise injection**: Simulate damaged tablets for robustness
5. **PDF extraction**: Use the 553MB publications.csv to extract more training pairs
6. **Curriculum learning**: Train on formulaic texts first, then complex ones

### Steps to Execute
1. [ ] Set up Kaggle API: `export KAGGLE_API_TOKEN=KGAT_...` from AIMO/.env
2. [ ] Download competition data: `kaggle competitions download -c deep-past-initiative-machine-translation`
3. [ ] Download external datasets (Akkademia, ORACC parallel corpus)
4. [ ] Build preprocessing pipeline
5. [ ] Create training notebook (Stage 1 + Stage 2)
6. [ ] Create inference notebook (Kaggle submission format)
7. [ ] Test locally, iterate on hyperparameters
8. [ ] Submit to Kaggle leaderboard
9. [ ] Evaluate and improve based on leaderboard feedback
