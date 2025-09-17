python scripts/generate_poison.py \
    --input datasets/clean/nq.jsonl \
    --output datasets/poisoned/nq_poisoned_10.jsonl \
    --poison_ratio 0.1

python scripts/generate_poison.py \
    --input datasets/clean/beir.jsonl \
    --output datasets/poisoned/beir_poisoned_10.jsonl \
    --poison_ratio 0.1


python scripts/generate_poison.py \
    --input datasets/clean/nq.jsonl \
    --output datasets/poisoned/nq_poisoned_30.jsonl \
    --poison_ratio 0.3

python scripts/generate_poison.py \
    --input datasets/clean/beir.jsonl \
    --output datasets/poisoned/beir_poisoned_30.jsonl \
    --poison_ratio 0.3


python scripts/generate_poison.py \
    --input datasets/clean/nq.jsonl \
    --output datasets/poisoned/nq_poisoned_50.jsonl \
    --poison_ratio 0.5

python scripts/generate_poison.py \
    --input datasets/clean/beir.jsonl \
    --output datasets/poisoned/beir_poisoned_50.jsonl \
    --poison_ratio 0.5
