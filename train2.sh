uv run scripts/compute_norm_stats.py --config-name pi0_lora_tacimg_tabero
uv run scripts/train.py pi0_lora_tacimg_tabero --exp-name=pi0_lora_tacimg_tabero_25 --overwrite
# 如需从头重跑同名实验，可以加上 --overwrite
#   --overwrite
uv run scripts/compute_norm_stats.py --config-name pi0_lora_tacfield_tabero
uv run scripts/train.py pi0_lora_tacfield_tabero --exp-name=pi0_lora_tacfield_tabero_25 --overwrite