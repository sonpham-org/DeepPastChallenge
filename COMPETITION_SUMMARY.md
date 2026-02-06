# Deep Past Challenge - Translate Akkadian to English

## Competition Overview
- **Host**: Deep Past Initiative (non-profit) on Kaggle
- **URL**: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation
- **Prize**: $50,000
- **Timeline**: Dec 16, 2025 → Mar 23, 2026 (entry deadline)
- **Task**: Build an AI model that translates 4,000-year-old Old Assyrian business records (cuneiform tablets) from Akkadian transliteration into English

## Historical Context
- ~22,000 clay tablets from ancient Kanesh (early 2nd millennium BCE)
- Contains contracts, letters, loans, receipts from Assyrian merchants
- Fewer than 20 living experts can read cuneiform
- Most tablets have not been read in 4,000 years

## Evaluation Metric
**Geometric Mean of BLEU and chrF++**:
```
Score = sqrt(BLEU × chrF++)
```
- **BLEU**: Word/phrase n-gram overlap precision (exact phrasing matters)
- **chrF++**: Character n-gram F-score with word boundaries (forgiving of morphological variants)
- Both computed via SacreBLEU
- If either metric is weak, overall score drops significantly

## Data Format

### train.csv (1,561 records, ~1.55 MB)
| Column | Description |
|--------|-------------|
| `oare_id` | Unique ID in OARE database |
| `transliteration` | Akkadian transliteration of the tablet |
| `translation` | English translation |

**CRITICAL**: Training data is **document-level** (full tablets), test data requires **sentence-level** predictions.

### test.csv (~4,000 sentences from ~400 documents)
| Column | Description |
|--------|-------------|
| `id` | Unique sentence identifier |
| `text_id` | Document identifier |
| `line_start` | Sentence boundary start |
| `line_end` | Sentence boundary end |
| `transliteration` | Akkadian text to translate |

### sample_submission.csv
| Column | Description |
|--------|-------------|
| `id` | Matches test.csv `id` |
| `translation` | English translation (one sentence) |

### Supplemental Data
| File | Size | Description |
|------|------|-------------|
| `published_texts.csv` | 10.79 MB | ~8,000 additional transliterations with metadata + AICC translations (poor quality) |
| `publications.csv` | 553.70 MB | ~880 OCR-extracted scholarly PDFs |
| `bibliography.csv` | small | Bibliographic metadata |
| `OA_Lexicon_eBL.csv` | 3.39 MB | 39,332 Akkadian vocabulary entries |
| `eBL_Dictionary.csv` | 1.55 MB | Akkadian dictionary from eBL |
| `Sentences_Oare_FirstWord_LinNum.csv` | - | Sentence alignment aid |

**Total dataset: ~600.95 MB**

## Transliteration Preprocessing Rules
- **Remove**: `!`, `?`, `/`, `:`, `.`, `˹`, `˺`
- **Keep content in**: `< >`, `<< >>`
- **Remove brackets but keep content**: `[ ]`
- **Text replacements**: `[x]` → `<gap>`, `...` → `<big_gap>`
- **Subscript numbers** → Unicode equivalents
- **Determinatives** remain in curly brackets: `{d}`, `{ki}`, `{lu₂}`, etc.

## Submission Rules
- Notebook-only submission (CPU or GPU)
- Maximum 9-hour runtime
- Internet access DISABLED during inference
- External data allowed but must be pre-loaded as Kaggle datasets
- Output: `submission.csv` with columns `id` and `translation`

## Key Statistics
- Source character length: 21-932 (avg 426.5)
- Source word count: 3-187 (avg 57.5)
- Akkadian vocabulary: ~74,114 tokens
- English vocabulary: ~46,588 tokens
- Length correlation (src/tgt): 0.817

## Known Approaches & Leaderboard Models

| Approach | Model | Notes |
|----------|-------|-------|
| ByT5 Two-Stage | `google/byt5-small` | Stage 1: general Akkadian, Stage 2: Old Assyrian fine-tuning |
| ByT5-base | `google/byt5-base` | Multiple notebooks, ensemble approaches |
| MarianMT | Helsinki-NLP MarianMT | Fine-tuned on Akkadian pairs |
| mBART-50 | `facebook/mbart-large-50` | Multilingual baseline |
| NLLB-200 | `facebook/nllb-200-distilled-600M` | Score ~23.2 on leaderboard |
| T5-base | `t5-base` | Standard NMT baseline |
| Custom Transformer | SentencePiece + 3-layer Transformer | Built from scratch |

## External Datasets Available
- **Akkademia corpus**: Pre-aligned Akkadian-English (~56K sentence pairs from ORACC)
- **ORACC Parallel Corpus**: On Kaggle, has `period` column for filtering by "Old Assyrian"
- **Extracted PDF pairs**: ~7,100 pairs from scholarly PDFs
- **praeclarum AICC corpus**: 130,000 AI-translated cuneiform texts

## Key GitHub Repos
1. [haakoan/Kaggle-Deep-Past-Challenge](https://github.com/haakoan/Kaggle-Deep-Past-Challenge-Translate-Akkadian-to-English) - ByT5 two-stage + NLLB
2. [kbsooo/Akkadian-to-English](https://github.com/kbsooo/Akkadian-to-English) - Multi-version with glossary augmentation
3. [Srism134/deep-past-akkadian-translation](https://github.com/Srism134/deep-past-akkadian-translation) - Custom Transformer

## Key Academic References
- Gutherz et al. (2023) - "Translating Akkadian to English with NMT" (PNAS Nexus) - CNN-based, BLEU4=37.47
- Lazar et al. (2021) - "Filling the Gaps in Ancient Akkadian Texts" (EMNLP)
- ByT5 for Cuneiform Lemmatization (ALP/NAACL 2025) - validates ByT5 superiority for cuneiform
