# flash


```
lerobot-train \
  --policy.type=diffusion
  --dataset.repo_id=/home/erik/flash/data/lerobot/cupstacking_notac \
  --dataset.path=/home/erik/flash/data/lerobot/cupstacking_notac \
  --output_dir=/home/erik/flash/src/imitator/outputs \
  --job_name=cupstacking_notac \
  --policy.repo_id=/home/erik/flash/data/lerobot/cupstacking_notac \
  --steps=60000
```

lerobot-train \
  --policy.type diffusion \
  --policy.use_separate_rgb_encoder_per_camera True \
  --dataset.root /home/erik/flash/data/lerobot/cupstacking \
  --dataset.repo_id local \
  --policy.repo_id local \
  --num_workers 8 \
  --steps 60000 \
  --wandb.enable True



lerobot-train \
  --dataset.root=/home/erik/flash/data/lerobot/cupstacking_notac \
  --dataset.repo_id=local \
  --policy.type=diffusion \
  --output_dir=/home/erik/flash/src/imitator/outputs/cupstacking_notac \
  --job_name=cupstacking_notac \
  --policy.device=cuda \
  --num_workers 16 \
  --steps 60000 \
  --wandb.enable=true \
  --policy.repo_id=local \
  --policy.push_to_hub=false