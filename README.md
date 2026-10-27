# IAHL-ReID

PyTorch implementation of IAHL-ReID on PRCC.

## Requirements

- Python 3.8, PyTorch 1.12.1, torchvision 0.13.1
- 1 × NVIDIA GPU

```bash
pip install -r requirements.txt
```

## Data

Request PRCC from the original authors and arrange it as:

```
DATASET_ROOT
└─ rgb
   ├── train
   ├── val
   └── test
       ├── A   # gallery
       ├── B   # same-clothes query
       └── C   # cloth-changing query
```

## Train

```bash
python main.py \
    --gpu_devices 0 \
    --dataset_root DATASET_ROOT \
    --dataset_filename rgb \
    --save_dir log/iahl_prcc \
    --save_checkpoint
```

## Evaluate

```bash
python main.py \
    --gpu_devices 0 \
    --dataset_root DATASET_ROOT \
    --dataset_filename rgb \
    --save_dir log/iahl_prcc \
    --resume log/iahl_prcc/best_mAP_model.pth \
    --evaluate
```

Evaluation reports mAP and Rank-{1, 5, 10} for both the same-clothes and
cloth-changing settings. Key options: `--num_parts` (K, default 4),
`--iahl_loss_weight` (λ, default 0.3), `--height/--width` (default 384 × 192).
