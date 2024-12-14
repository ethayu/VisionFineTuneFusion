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
python3 depth_estimation/train_depth.py \
    --image_dir data/depth/images \
    --depth_dir data/depth/depth_maps \
    --model_type dino
```

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
python3 depth_estimation/evaluate_depth.py \
    --image_dir data/depth/images \
    --depth_dir data/depth/depth_maps \
    --checkpoint_path depth_estimation/checkpoints/best_model.pth \
    --save_predictions \
    --results_file evaluation_results.json
```
---

### 3. Metrics

The following metrics are calculated during evaluation:
- **Mean Absolute Error (MAE)**: Average absolute pixel-wise error between predicted and ground truth depth maps.
- **Root Mean Squared Error (RMSE)**: Square root of the mean squared error between predicted and ground truth depth maps.
