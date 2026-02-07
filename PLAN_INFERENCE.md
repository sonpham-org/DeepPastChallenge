# Inference Improvement Plan

## Current Baseline
- Model: ByT5-small, two-stage fine-tuned
- Stage 1 best: BLEU=14.23, chrF++=26.07, Combined=19.26
- Stage 2 best: BLEU=7.76, chrF++=25.45, Combined=14.05
- Inference: beam search (width=4), repetition_penalty=1.2

## Competition Metric
`sqrt(BLEU * chrF++)` — geometric mean, punishes if either is weak.
BLEU is our bottleneck (~7.5 vs chrF++ ~25).

## Daily Budget: 5 submissions/day

## Inference Techniques (No Retraining Needed)

### Tier 1: High Impact
1. **MBR Decoding** (N=16-32 samples, chrF or custom utility)
   - Expected: +2-4 BLEU
   - Libraries: ZurichNLP/mbr, fastChrF
2. **Custom MBR utility** matching sqrt(BLEU*chrF++) directly

### Tier 2: Free / Easy Wins
3. **no_repeat_ngram_size=3** — fix repetition problem
4. **length_penalty tuning** (sweep 0.6-1.2)
5. **num_beams tuning** (try 5-8)
6. **Post-processing**: collapse repeated phrases, unicode normalization

### Tier 3: Needs Training Changes
7. **Checkpoint averaging** (save multiple, average weights)
8. **Ensemble multiple models** (different seeds, hyperparams)

## Suggested Daily Submission Plan
1. Beam search tuning (no_repeat_ngram, length_penalty sweep)
2. MBR with chrF utility (N=16)
3. MBR with custom sqrt(BLEU*chrF++) utility (N=16)
4. MBR with N=32 (more samples)
5. Best config + post-processing

## TODO
- [ ] Research more techniques (full sweep)
- [ ] Implement MBR decoding in inference notebook
- [ ] Implement beam search parameter sweep
- [ ] Implement post-processing pipeline
