# Physics-Informed Attention U-Net (PIAUNet): An Enhanced U-Net Framework for Underwater Segmentation in Aquaculture

This repository contains the implementation of **PIAU-Net (Physics-Informed Attention U-Net)**, a custom image segmentation framework developed in **PyTorch**.  
The model extends the classical U-Net architecture by explicitly integrating **optical physics priors** into both the **network architecture** and the **training objective**.

This project is a part of project sponsored under Ministry of Earth Science title ``Development of Automated Underwater Species Identification System Using Deep Learning Techniques".

---

## Overview

Standard encoder–decoder segmentation networks rely purely on appearance-based learning. In many real-world optical imaging scenarios, however, image formation is governed by physical processes such as light attenuation, scattering, and backscatter. Ignoring these effects often leads to unstable predictions under varying illumination.

PIAU-Net addresses this limitation by embedding **physics awareness** directly into pipeline through:

- A **Physics Branch** that learns physically meaningful cues
- **Physics-Informed Attention Gates (PAGs)** that regulate skip connections
- A **Physics-Guided Loss Function** that enforces physical consistency during training

The result is a segmentation model that is more robust, interpretable, and illumination-aware than conventional U-Net variants.

---

## Key Contributions

- **Physics-Informed Architecture**  
  A modified U-Net that integrates physics-derived feature maps into the decoding process via attention gating.

- **Physics-Aware Attention Gates (PAGs)**  
  Skip connections are selectively filtered using physics-based cues, suppressing unreliable features caused by illumination distortions.

- **Physics-Guided Loss Function**  
  Training is regularized using a physics-consistency term in addition to standard segmentation loss, improving stability and boundary accuracy.

- **Clean and Modular Design**  
  The code is structured for clarity and easy extension, making it suitable for academic and thesis use.

---

## Model Architecture

PIAU-Net follows an encoder–decoder design with skip connections, similar to U-Net, with the following enhancements:

![PIAUNet Architecture](./Architecture.jpg)   
- **Encoder**  
  Extracts hierarchical features using convolutional layers and max pooling.

- **Bottleneck**  
  Captures global contextual information.

- **Physics Branch**  
  Processes bottleneck features to generate physics-inspired representations (e.g., visibility- or attenuation-like cues).
  ![Physics Branch Architecture](PB_Architecture.png)   

- **Physics-Informed Attention Gates**  
  Use physics features as gating signals to refine skip-connection information before fusion with the decoder.

- **Decoder**  
  Reconstructs spatial details using upsampling and gated skip connections.

- **Output Layer**  
  Produces a single-channel output followed by a sigmoid activation for binary segmentation.

---

## Physics-Guided Learning

### Physics-Aware Attention Gates
The attention mechanism is not purely data-driven. Physics-derived features guide the gating process to:

- Suppress features from low-visibility or physically inconsistent regions
- Emphasize regions with reliable optical information
- Improve boundary localization under illumination changes

### Physics-Guided Loss Function
The total training loss combines:

- **Segmentation Loss** (e.g., Binary Cross-Entropy + Dice-based loss)
- **Physics Consistency Loss**, which regularizes predictions to remain consistent with learned physical cues

This coupling encourages illumination-invariant and physically meaningful segmentation outputs.

---

## Dataset Assumptions

The current codebase assumes a **binary segmentation dataset** consisting of:

- RGB input images
- Corresponding binary masks

Typical directory structure:

dataset/
├── images/
│ ├── sample_001.png
│ └── sample_002.png
└── masks/
├── sample_001.png
└── sample_002.png

Mask values:
- `0` → Background
- `1` → Foreground

---

## Training Configuration

- **Output Channels:** 1  
- **Activation:** Sigmoid  
- **Loss Function:**  
  - Binary segmentation loss  
  - Physics-guided regularization term  
- **Optimizer:** Adam / AdamW  
- **Learning Rate:** Typically `1e-4`  
- **Evaluation Metrics:** Dice, IoU, Precision, Recall  

The training pipeline supports validation monitoring and checkpoint saving.

---

## Running Training

```bash
python train.py
```

Training parameters such as batch size, learning rate, and number of epochs can be adjusted within the training script.

## Inference

During inference:

- The network outputs a probability map in the range [0, 1]

- A threshold (e.g., 0.5) is applied to obtain the final binary mask

- The model can be used as a standalone segmenter or as a preprocessing module for downstream tasks.

## Dependencies

Required Python packages:
```bash
pip install torch torchvision numpy opencv-python albumentations tqdm
```

A CUDA-enabled GPU is recommended for efficient training.

## Extensibility

Although this repository focuses on binary segmentation, the framework is designed to be extensible to:

- Multi-class segmentation

- ROI generation for object detection

- Additional physics-based constraints and losses

Minimal refactoring is required to adapt the architecture to new datasets or tasks.