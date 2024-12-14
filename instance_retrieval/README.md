# Instance Retrieval Baseline

This module implements an **instance retrieval baseline** using **DiNO** (or DiNO + Autoencoder) as a feature extractor. The goal is to retrieve images similar to a query image from a dataset, leveraging the feature representations extracted by DiNO.

---

## Folder Structure
```markdown
instance_retrieval/ 
├── extract_features.py # Script to extract features using DiNO 
├── build_index.py # Script to build a FAISS index 
├── query_index.py # Script to query the FAISS index 
├── metrics.py # Metrics for evaluating instance retrieval 
├── init.py # Makes this a Python package
```

---

## Features

1. **Feature Extraction**:
   - Extract features from RGB images using **DiNO** or **DiNO + Autoencoder**.
   - Save features and associated metadata for efficient retrieval.

2. **FAISS Indexing**:
   - Use **FAISS** to create a fast nearest-neighbor index for the extracted features.

3. **Query and Retrieval**:
   - Query the FAISS index with an image to find similar images based on feature embeddings.

4. **Evaluation**:
   - Evaluate retrieval performance using metrics such as:
     - **Recall@K**
     - **Mean Average Precision (mAP)**

---

## Dataset Preparation

### Oxford 102 Flowers Dataset

1. **Download the Dataset**:
   - Visit the [Oxford 102 Flowers dataset page](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
   - Download the **image dataset** and **labels.mat** file.

2. **Extract the Dataset**:
   - Place the images in the `data/flowers/images` folder.
   - Place the `labels.mat` file in the `data/flowers/` folder.

3. **Organize the Dataset**:
```markdown
data/flowers/ ├── images/ # RGB images ├── labels.mat # Label metadata
```

---

## Usage

### 1. Extract Features

Run the `extract_features.py` script to extract features from the dataset:
```bash
python extract_features.py
```

**Key arguments**:
- `image_dir`: Path to the directory containing images.
- `output_file`: Path to save the extracted features.
- `model_type`: `"dino"` for baseline DiNO or `"custom"` for DiNO + Autoencoder.
- `autoencoder_path`: Path to trained autoencoder weights (only for `"custom"` model type).

Example:
```bash
python extract_features.py
--image_dir data/flowers/images
--output_file instance_retrieval/features.npz
--model_type dino
```

---

### 2. Build the FAISS Index

Run the `build_index.py` script to build a FAISS index:
```bash
python build_index.py
```

**Key arguments**:
- `features_file`: Path to the `.npz` file containing extracted features.
- `index_file`: Path to save the FAISS index.

Example:
```bash
python build_index.py
--features_file instance_retrieval/features.npz
--index_file instance_retrieval/faiss_index.idx
```

---

### 3. Query the Index

Run the `query_index.py` script to retrieve similar images:
```bash
python query_idnex.py
```

**Key arguments**:
- `query_image_path`: Path to the query image.
- `features_file`: Path to the `.npz` file containing extracted features.
- `index_file`: Path to the FAISS index file.
- `model_type`: `"dino"` for baseline DiNO or `"custom"` for DiNO + Autoencoder.
- `autoencoder_path`: Path to trained autoencoder weights (only for `"custom"` model type).

Example:
```bash
python query_index.py
--query_image_path data/flowers/images/image_0001.jpg
--features_file instance_retrieval/features.npz
--index_file instance_retrieval/faiss_index.idx
--model_type dino
```

---

### 4. Evaluate the Retrieval

Run the `metrics.py` functions to compute retrieval metrics:
- **Recall@K**
- **Mean Average Precision (mAP)**

Example script to evaluate:
```bash
python3 instance_retrieval/metrics.py \
  --index_file instance_retrieval/faiss_index.idx \
  --features_file instance_retrieval/features.npz \
  --top_k 100 \
  --num_queries 50 \
  --similarity_threshold 0.8 \
  --output_file results/evaluation_results.json

---

## Metrics

The following metrics are calculated during evaluation:
- **Recall@K**: Proportion of relevant images retrieved within the top \( K \).
- **Mean Average Precision (mAP)**: Evaluates ranking quality of retrieved images.

---

## Notes

1. **Dataset Compatibility**: The code is designed to work with any dataset containing RGB images.
2. **Modularity**: You can swap in your custom models (e.g., DiNO + Autoencoder) for feature extraction.
3. **Extensibility**: The scripts can be extended to handle multi-modal retrieval tasks.

---

## References

- [Oxford 102 Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [DiNOv2 GitHub Repository](https://github.com/facebookresearch/dinov2)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
