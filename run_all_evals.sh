#!/bin/bash

# ===== CLEAN =====
python3 run_eval.py --dataset nq   --poison_ratio 0 --topk 5 --zkip 0 --gen_model gpt-4o-mini     --out results/nq_clean.jsonl
python3 run_eval.py --dataset beir --poison_ratio 0 --topk 5 --zkip 0 --gen_model gpt-4o-mini     --out results/beir_clean.jsonl

# ===== NQ POISONED =====
for r in 0.1 0.2 0.3; do
  python3 run_eval.py --dataset nq --poison_ratio $r --topk 5 --zkip 0      --gen_model gpt-4o-mini --out results/nq_poison_${r}.jsonl
done

# ===== BEIR POISONED =====
for r in 0.1 0.2 0.3; do
  python3 run_eval.py --dataset beir --poison_ratio $r --topk 5 --zkip 0      --gen_model gpt-4o-mini --out results/beir_poison_${r}.jsonl
done

# ===== NQ WITH ZKIP =====
for r in 0.1 0.2 0.3; do
  python3 run_eval.py --dataset nq --poison_ratio $r --topk 5 --zkip 1      --gen_model gpt-4o-mini --out results/nq_poison_${r}_zkip.jsonl
done

# ===== BEIR WITH ZKIP =====
for r in 0.1 0.2 0.3; do
  python3 run_eval.py --dataset beir --poison_ratio $r --topk 5 --zkip 1      --gen_model gpt-4o-mini --out results/beir_poison_${r}_zkip.jsonl
done

# ===== MULTI-RUN FOR STATISTICS (SEEDS) =====
for seed in 42 123 999; do
  python3 run_eval.py --dataset nq   --poison_ratio 0.1 --topk 5 --zkip 1 --seed $seed --gen_model gpt-4o-mini --out results/nq_seed_${seed}.jsonl
  python3 run_eval.py --dataset beir --poison_ratio 0.1 --topk 5 --zkip 1 --seed $seed --gen_model gpt-4o-mini --out results/beir_seed_${seed}.jsonl
done

