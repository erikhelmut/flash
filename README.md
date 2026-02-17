# flash



lerobot-train \
  --dataset.root=/home/erik/flash/data/lerobot/cupstacking_v2 \
  --dataset.repo_id=local \
  --policy.type=diffusion \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --output_dir=/home/erik/flash/src/imitator/outputs/cupstacking_v2 \
  --job_name=cupstacking \
  --policy.device=cuda \
  --num_workers 16 \
  --steps 60000 \
  --wandb.enable=true \
  --policy.repo_id=local \
  --policy.push_to_hub=false

lerobot-dataset-viz --repo-id /home/erik/flash/src/imitator/outputs/cupstacking --root /home/erik/flash/src/imitator/outputs/cupstacking --mode local --episode-index 2 --display-compressed-images true