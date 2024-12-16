# VisionFineTuneFusion: Efficient Vision-Language Alignment for Retrieval and Depth Estimation

This project focuses on integrating DINO's structural embeddings with CLIP's semantic insights to enhance various vision tasks, demonstrating improved performance in **instance retrieval** and **depth estimation** with a more compact embedding dimension (512 vs. 768). Our approach introduces a novel framework that fine-tunes DINO embeddings using lightweight autoencoders while aligning them with CLIP embeddings to inject semantic information. The method is efficient, leveraging frozen pre-trained models (DINO and CLIP) and adding task-specific learning through reconstruction and alignment losses.

## Approach

- **Two Autoencoders**: We train two autoencoders, one for DINO's class token (CLS) embedding and another for its patch embeddings. These autoencoders compress the embeddings while preserving reconstruction fidelity and semantic alignment.
- **Alignment with CLIP**: During training, we align DINO's embeddings with CLIP's multimodal embeddings using cosine similarity loss. This alignment is applied to both the CLS and patch embeddings.
- **Frozen Pretrained Models**: Both DINO and CLIP weights are frozen, ensuring that the training focuses on learning lightweight, task-specific embeddings.

## Tasks

1. **Instance Retrieval**:
   - Retrieve images similar to a query image based on their visual features.
   - Uses DINO’s CLS and patch embeddings as input, fine-tuned with CLIP alignment.
   - Efficient indexing and retrieval are implemented using FAISS.

2. **Depth Estimation**:
   - Predict pixel-wise depth maps from RGB images.
   - Uses DINO’s features as input to two separate autoencoders (CLS and patch-based).
   - Outputs compressed, semantically-enriched embeddings used for regression tasks.

## Training Pipeline

1. **Embedding Extraction**:
   - Pass the image through DINO to obtain its CLS and patch embeddings.
   - Pass the image and corresponding text through CLIP to extract two global embeddings (image and text).
   - Patch the image, upsample each patch, and pass through CLIP to obtain patch-wise embeddings.

2. **Autoencoder Training**:
   - Train the CLS autoencoder with reconstruction loss and a CLIP alignment loss using CLIP’s global embeddings.
   - Train the patch autoencoder with reconstruction loss and alignment loss for each patch embedding with CLIP.

3. **Loss Function**:
   - The total loss combines reconstruction and alignment objectives:
     ```
     L_total = L_reconstruction^CLS + L_alignment^CLS + L_reconstruction^patch + L_alignment^patch
     ```

## Results

- **Instance Retrieval**: Achieved higher retrieval accuracy with smaller embedding dimensions (512 vs. 768).
- **Depth Estimation**: Improved depth mapping results while leveraging compressed embeddings, enhancing both global (CLS) and local (patch) predictions.
---

## Folder Structure
```markdown
visionfinetunefusion/
├── data/
│   ├── raw/                    # Raw image-text datasets (e.g., COCO, LAION)
│   ├── processed/              # Preprocessed data ready for training
│   └── dataset_loader.py       # Code for loading and preprocessing datasets
├── models/
│   ├── dino_model.py           # Code to load and interface with DiNOv2
│   ├── clip_model.py           # Code to load and interface with CLIP
│   ├── autoencoder.py          # Implementation of the autoencoder
│   └── losses.py               # Implementation of loss functions
├── training/
│   ├── train.py                # Script to train the autoencoder
│   ├── train_config.yaml       # Config file for training hyperparameters
│   ├── evaluation.py           # Evaluation script for downstream tasks
│   └── scheduler.py            # Learning rate scheduler and optimizer setup
├── scripts/
│   ├── preprocess.py           # Preprocessing script for raw data
│   ├── visualize_latent.py     # Script to visualize embeddings in latent space
│   └── inference.py            # Script to run inference on new data
├── utils/
│   ├── logger.py               # Logging utilities for training and debugging
│   ├── metrics.py              # Functions to compute metrics like cosine similarity
│   └── helpers.py              # Miscellaneous utility functions
│── results/
│   ├── checkpoints/            # Directory to save model checkpoints
│   ├── logs/                   # Training and evaluation logs
│   └── plots/                  # Plots of results, embeddings, etc.
├── instance_retrieval/              # Instance retrieval baseline
│   ├── extract_features.py          # Extract features using DiNO
│   ├── build_index.py               # Build a FAISS index
│   ├── query_index.py               # Query the FAISS index
│   ├── metrics.py                   # Evaluate retrieval performance
│   ├── README.md                    # Documentation for instance retrieval
│   └── __init__.py                  # Makes this a Python package
├── depth_estimation/                # Depth estimation baseline
│   ├── train_depth.py               # Train the depth estimation model
│   ├── evaluate_depth.py            # Evaluate the depth estimation model
│   ├── metrics.py                   # Evaluation metrics
│   ├── model.py                     # Depth estimation architecture
│   ├── README.md                    # Documentation for depth estimation
│   └── __init__.py                  # Makes this a Python package
├── requirements.txt                 # Python dependencies
├── README.md                        # Root README for the project
└── .gitignore                       # Git ignore file for unnecessary files
```
---

## Dataset Preparation

### Download COCO Dataset

1. **Dataset Overview**:
   - The [COCO (Common Objects in Context)](https://cocodataset.org/) dataset contains images of complex everyday scenes with their corresponding annotations, including bounding boxes and captions.
   - We will use the **2017 Train and Validation splits**.

2. **Download COCO 2017 Images**:
   - Train images: [2017 Train Images (118K)](http://images.cocodataset.org/zips/train2017.zip)
   - Validation images: [2017 Val Images (5K)](http://images.cocodataset.org/zips/val2017.zip)

3. **Download COCO 2017 Annotations**:
   - Train/Val annotations: [2017 Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

4. **Extract the Files**:
   - Extract the downloaded files into the `data/coco/` directory:
     ```markdown
     data/
     ├── train2017/             # Training images
     ├── val2017/               # Validation images
     ├── annotations/           # Annotations for both splits
     ```

5. **Preprocess Images**:
   - Resize images to 224x224 if necessary for model compatibility.

---

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ethayu/VisionFineTuneFusion.git
cd VisionFineTuneFusion
```

### 2. Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Instance Retrieval

Refer to the [`instance_retrieval/README.md`](./instance_retrieval/README.md) for details.

Steps:
1. Extract features from COCO images.
2. Build a FAISS index using the extracted features.
3. Query the index with sample images to retrieve similar images.

---

### 2. Depth Estimation

Refer to the [`depth_estimation/README.md`](./depth_estimation/README.md) for details.

Steps:
1. Train the depth estimation model on COCO.
2. Evaluate the model on the validation set using depth metrics such as MAE and RMSE.

---

## References

- [COCO Dataset](https://cocodataset.org/)
- [DiNOv2 GitHub Repository](https://github.com/facebookresearch/dinov2)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [NYU Depth V2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

---

## Future Work

This project serves as a baseline for instance retrieval and depth estimation tasks. Future extensions could include:
- Adding tasks such as object detection and segmentation.
- Evaluating performance on other datasets like KITTI or NYU Depth V2.
- Exploring different loss functions and architectures for fine-tuning DiNO.

Feel free to contribute!
