# DeepPastChallenge - Claude Code Instructions

## Project
Kaggle competition: Translate 4,000-year-old Akkadian cuneiform to English.
- URL: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation
- Prize: $50K, Deadline: Mar 23, 2026
- Metric: `sqrt(BLEU * chrF++)` via SacreBLEU

## Approach
ByT5 two-stage fine-tuning:
1. Stage 1: General Akkadian (Akkademia 50K + ORACC 2K + competition 1.5K)
2. Stage 2: Old Assyrian specialization (competition data only)

## Files
- `train_byt5.ipynb` - Training notebook (run on Kaggle GPU)
- `inference_byt5.ipynb` - Inference notebook (Kaggle submission)
- `COMPETITION_SUMMARY.md` - Full competition details
- `PLAN.md` - Implementation plan
- `data/` - Competition data (gitignored)
- `external_data/` - Akkademia + ORACC corpora (gitignored)

## Kaggle CLI
```bash
export $(grep KAGGLE_API_TOKEN /home/son/GitHub/AIMO/.env | xargs)
```

## Key Constraints
- Notebook-only submission, 9hr max runtime, no internet during inference
- Train data is document-level, test data is sentence-level
- Must upload trained model as Kaggle dataset for inference notebook
