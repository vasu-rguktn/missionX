# MobileNetV2 Explainable AI (XAI) Documentation

This document provides a detailed explanation of the code and the insights drawn from the Jupyter Notebook `V2_XAI_Methods.ipynb`. The notebook focuses on evaluating a fine-tuned MobileNetV2 model for Pneumonia detection and utilizing Explainable AI techniques, specifically GradCAM, to understand the model's decision-making process.

## 1. Setup and Dependencies

The initial section of the notebook is dedicated to setting up the environment. It installs essential libraries required for model evaluation and interpretability.

```python
!pip install lime shap torchcam opencv-python matplotlib
```

The core libraries used include:
*   `torch`, `torchvision`: For working with the PyTorch model and image transformations.
*   `torchcam`: Specifically for implementing Gradient-weighted Class Activation Mapping (GradCAM).
*   `cv2` (OpenCV), `PIL`, `matplotlib`: For core image processing and visualization of the heatmaps.

## 2. Model Configuration and Loading

The notebook utilizes the MobileNetV2 architecture. The model is prepared for inference as follows:

```python
import torch
import torch.nn as nn
from torchvision import models

# Initialize MobileNetV2 without pre-trained ImageNet weights
model = models.mobilenet_v2(pretrained=False)

# Modify the final classifier layer for binary classification
# (e.g., normal vs. pneumonia)
model.classifier[1] = nn.Linear(model.last_channel, 2)

# Load the fine-tuned model weights
MODEL_PATH = "/content/drive/MyDrive/mobilenet_pneumonia.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval() # Set to evaluation mode
```

This setup modifies the network's head to output two classes and loads custom trained weights, ensuring the model is ready to make predictions on X-ray images.

## 3. Image Preprocessing Pipeline

Before feeding an image to the model, it undergoes a series of standard transformations. 

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
These steps guarantee that the input matches the format expected by the MobileNetV2 architecture (standard resizing and ImageNet-based normalization).

## 4. Prediction Mechanism

The notebook defines a robust `predict_image` function to handle individual image predictions:

*   It opens the image using `PIL.Image`.
*   Applies the predefined `transform` pipeline.
*   Uses `torch.no_grad()` to perform an inference pass safely and efficiently on the specific `device`.
*   Calculates probability utilizing `torch.softmax` on the model outputs.
*   Returns the human-readable predicted label (`Normal` or `Pneumonia`) along with its confidence score.

## 5. Batch Evaluation and Critical Insights

A significant portion of the notebook performs batch predictions over a folder of test images, specifically iterating through the `NORMAL` class directory.

**Crucial Observation & Insight:**
When executing the batch tests on images known to be "NORMAL", the execution output reveals a severe, consistent misclassification issue. 

*   **Observation:** Almost all images labeled as `NORMAL` (e.g., `IM-0001-0001.jpeg`, `IM-0005-0001.jpeg`, etc.) are predicted by the model as **`Pneumonia`**.
*   **Confidence:** Alarmingly, the confidence scores for these incorrect predictions are extremely high, frequently hitting **100.00%**.

**Implications:** 
This insight suggests a major flaw in the model's training process. It indicates that the model has not generalized well and is suffering from severe bias. Possible root causes that require investigation include:
1.  **Extreme Class Imbalance:** The training dataset might have contained significantly more "Pneumonia" cases than "Normal" cases, causing the model to default to the majority class.
2.  **Data Leakage/Artifacts:** The model may be fixated on a specific artifact present only in the "Pneumonia" scans (e.g., a specific hospital stamp, crop pattern, or device mark) rather than true anatomical features.
3.  **Flawed Data Labeling:** The training or testing dataset might be incorrectly labeled or structured.

## 6. Explainable AI (XAI) Techniques

To investigate *why* the model is making its predictions (and specifically why it's failing on Normal cases), a variety of Explainable AI (XAI) techniques are introduced. These tools provide different perspectives on the model's inner workings.

### 6.1 GradCAM & GradCAM++
**Gradient-weighted Class Activation Mapping (GradCAM) and its enhanced version (GradCAM++)** compute activation maps that highlight the areas of the X-ray most influential to the model's prediction.

*   **Initialization:** 
    ```python
    from torchcam.methods import GradCAM # or GradCAMpp
    
    # Target a specific late-stage convolutional layer
    cam_extractor = GradCAM(model, target_layer="features.18")
    ```
    The `features.18` layer in MobileNetV2 represents high-level abstract features just before the classification head. While standard GradCAM averages gradients, GradCAM++ uses a weighted combination of positive partial derivatives for finer-grained visualizations.
*   **The Visualization Process (`show_gradcam` function):** Enables gradient tracking (`img.requires_grad_()`) and extracts the map using the `cam_extractor`. OpenCV generates a heatmap overlaid on the original X-ray.

### 6.2 SHAP (SHapley Additive exPlanations)
**SHAP** is a game-theoretic approach to explain the output of the model by assigning each region (or super-pixel) of the image an importance value.

*   **Process:** SHAP creates interpretations by evaluating predictions across different subsets of the image dimensions. An explainer breaks the input X-ray into super-pixels and uses deep explainer models tailored for PyTorch.
*   **Insight:** SHAP outputs a visualization explicitly distinguishing which pixels forcefully pushed the likelihood toward "Pneumonia" versus "Normal." This uncovers potential artifact-based bias (like medical text or equipment overlays on the X-ray) that the model might be inappropriately exploiting.

### 6.3 LIME (Local Interpretable Model-agnostic Explanations)
**LIME** provides interpretability by training an interpretable surrogate model locally around a specific prediction.

*   **Process:** LIME generates numerous random perturbations of the input X-ray (by making various super-pixels gray or blank) and observes how MobileNetV2 changes its confidence.
*   **Insight:** It isolates the exact lung segments or boundaries that most heavily contributed to the 100% "Pneumonia" prediction. Acting as a model-agnostic technique, LIME offers a reliable check against gradient-based insights like GradCAM.

By combining these visualization methodologies, developers can comprehensively audit the misclassified "Normal" images to debug the systematic biases learned during model training.
