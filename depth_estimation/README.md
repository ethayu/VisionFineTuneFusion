# Depth Estimation Baseline

This module implements a **depth estimation baseline** using **DiNO** (or DiNO + Autoencoder) as a feature extractor. The goal is to train and evaluate a model that predicts depth maps from RGB images.

---

## Folder Structure
```markdown
depth_estimation/ 
├── train_depth.py # Script to train the depth estimation model 
├── evaluate_depth.py # Script to evaluate the depth estimation model 
├── metrics.py # Metrics for evaluating depth estimation 
├── model.py # Define depth estimation architecture 
├── init.py # Makes this a Python package
```

---

## Features

1. **Feature Extraction**:
   - Uses **DiNOv2** or a **DiNO + Autoencoder** model to extract meaningful features from RGB images.
   - These features are passed to a regression head for depth prediction.

2. **Training**:
   - Trains the regression head using pixel-wise loss functions like Mean Squared Error (MSE).
   - Supports datasets with paired RGB images and depth maps.

3. **Evaluation**:
   - Evaluates the model using common depth estimation metrics such as:
     - **Mean Absolute Error (MAE)**
     - **Root Mean Squared Error (RMSE)**

---
## Dataset Preparation

### NYU Depth V2 Dataset

1. **Download the Dataset**:
   - Visit the [NYU Depth V2 Dataset page](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).
   - Download the preprocessed dataset from [this Google Drive link](https://drive.google.com/drive/folders/1L6ndgDvnC1BXo9dwoERsahC6vD5CCO3J).

2. **Extract the Dataset**:
   - Unzip the dataset into the `data/depth/` folder.

```markdown
data/depth/ 
├── images/ # RGB images 
├── depth_maps/ # Corresponding depth maps
```

3. **Ensure Consistent Resolutions**:
- Resize both RGB images and depth maps to the same resolution (e.g., 224x224).




---

## Usage

### 1. Train the Model

Run the `train_depth.py` script to train the model:
```bash
python train_depth.py
```

**Key arguments**:
- `image_dir`: Path to the directory with RGB images.
- `depth_dir`: Path to the directory with depth maps.
- `model_type`: `"dino"` for baseline DiNO or `"custom"` for DiNO + Autoencoder.
- `autoencoder_path`: Path to trained autoencoder weights (only for `"custom"` model type).

Example:
```bash

For DINO model:

python3 depth_estimation/train_depth.py \
    --image_dir data/depth/images \
    --depth_dir data/depth/depth_maps \
    --model_type dino \
    --checkpoint_dir depth_estimation/checkpoints \
    --epochs 10 \
    --batch_size 8 \
    --lr 0.03

For custom_patch model 

python3 depth_estimation/train_depth.py \
    --image_dir data/depth/images \
    --depth_dir data/depth/depth_maps \
    --model_type custom_patch \
    --patch_autoencoder_path depth_estimation/patch_model.pth \
    --checkpoint_dir depth_estimation/checkpoints \
    --epochs 10 \
    --batch_size 8 \
    --lr 0.03

For custom_cls model:

python3 depth_estimation/train_depth.py \
    --image_dir data/depth/images \
    --depth_dir data/depth/depth_maps \
    --model_type custom_cls \
    --cls_autoencoder_path depth_estimation/cls_model.pth \
    --checkpoint_dir depth_estimation/checkpoints \
    --epochs 10 \
    --batch_size 8 \
    --lr 0.03


---

### 2. Evaluate the Model

Run the `evaluate_depth.py` script to compute evaluation metrics:

```bash
python evaluate_depth.py
```

**Key arguments**:
- `image_dir`: Path to the directory with RGB images.
- `depth_dir`: Path to the directory with depth maps.
- `model_checkpoint`: Path to the trained model checkpoint.

Example:
```bash
For DINO model:

python3 depth_estimation/evaluate_depth.py \
    --image_dir data/depth/images \
    --depth_dir data/depth/depth_maps \
    --model_type dino \
    --checkpoint_path depth_estimation/checkpoints/best_model_dino.pth \
    --save_predictions \
    --results_file results/dino_depth.json

For custom_patch model:

python3 depth_estimation/evaluate_depth.py \
    --image_dir data/depth/images \
    --depth_dir data/depth/depth_maps \
    --model_type custom_patch \
    --checkpoint_path depth_estimation/checkpoints/checkpoint_epoch_10.pth \
    --patch_autoencoder_path depth_estimation/patch_model.pth \
    --save_predictions \
    --results_file results/patch_depth.json

For custom_cls model:

python3 depth_estimation/evaluate_depth.py \
    --image_dir data/depth/images \
    --depth_dir data/depth/depth_maps \
    --model_type custom_cls \
    --checkpoint_path depth_estimation/checkpoints/best_model.pth \
    --cls_autoencoder_path depth_estimation/cls_model.pth \
    --save_predictions \
    --results_file results/cls_depth.json
    
```
---

Visualize.py:

python3 depth_estimation/visualize.py \
    --image_path data/depth/images/00000.jpg \
    --depth_dir data/depth/depth_maps \
    --checkpoint_path depth_estimation/checkpoints/best_model.pth \
    --model_type dino \
    --output_path results/dino_depth_vis.png

For CLS autoencoder:

python3 depth_estimation/visualize.py \
    --image_path data/depth/images/00005.jpg \
    --depth_dir data/depth/depth_maps \
    --checkpoint_path depth_estimation/checkpoints/best_model.pth \
    --model_type custom_cls \
    --cls_autoencoder_path depth_estimation/cls_model.pth \
    --output_path results/cls_depth_vis.png

For PatchAutoencoder:

python3 depth_estimation/visualize.py \
    --image_path data/depth/images/01000.jpg \
    --depth_dir data/depth/depth_maps \
    --checkpoint_path depth_estimation/checkpoints/best_model.pth \
    --model_type custom_patch \
    --patch_autoencoder_path depth_estimation/patch_model.pth \
    --output_path results/patch_depth_vis.png









