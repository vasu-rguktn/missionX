# MobileNetV2 Performance Evaluation

This document outlines the evaluation process, methodologies, and performance metrics gathered in the `v2_Metrics.ipynb` notebook for the fine-tuned MobileNetV2 model on the chest X-ray dataset.

## 1. Environment and Data Setup
- **Dependencies Installed:** The evaluation environment relies on `scikit-learn` for generating detailed evaluation metrics and `seaborn` for clear visualizations, alongside standard libraries like PyTorch, Torchvision, NumPy, and Matplotlib.
- **Image Preprocessing:** Test datasets undergo a strict transformation pipeline to match the conditions the model was trained on:
  - Images are resized to **224x224**.
  - While essentially X-rays, images are processed as **Grayscale with 3 output channels** (to fit the MobileNetV2 expected architecture).
  - Tensors are constructed and normalized with a mean of 0.5 and standard deviation of 0.5 across all three channels.
- **Test Set Loading:** The data is pulled directly from the test directory using PyTorch's `ImageFolder`. A `DataLoader` is created to iterate over the test samples in batches of 32 without shuffling to maintain their inherent order for evaluation.

## 2. Model Initialization and Preparation
- **Architecture Setup:** An instance of MobileNetV2 is instantiated without default pretrained weights.
- **Custom Classifier:** The final fully connected layer of the model's classifier is explicitly modified from its standard configuration to a Linear layer outputting exactly **2 classes** (representing the binary classification task).
- **Loading Checkpoint:** The custom weights from a saved model state (`mobilenet_pneumonia.pth`) are loaded into the initialized architecture.
- **Inference Mode:** The model is transferred to the designated compute device (GPU if available, else CPU) and set entirely to evaluation mode to prevent any weight updates and to ensure layers like Dropout act appropriately.

## 3. Prediction Generation
The notebook processes the test data in a loop:
- It iterates through the entire test `DataLoader`.
- Gradient calculation is temporarily disabled (`torch.no_grad()`) to significantly reduce memory footprint and increase inference speed.
- The model computes output probabilities, and the index of the highest probability is taken as the designated `predicted` class.
- Two primary lists, `y_true` (actual labels) and `y_pred` (model predictions), are accumulated across all test images to perform a holistic evaluation.

## 4. Comprehensive Metric Analysis
Beyond calculating the raw overall accuracy score, the notebook dives deep into robust classification metrics to understand the model's actual effectiveness:

### A. Summary Classification Report
The notebook leverages `classification_report` to break down performance across both classes. This yields critical diagnostic numbers including:
- **Precision:** Measuring the model's exactness (when it predicts a certain class, how often is it correct?).
- **Recall (Sensitivity):** Measuring the model's completeness (out of all the actual instances of a class, how many did it successfully find?).
- **F1-Score:** The harmonic mean between precision and recall, useful for understanding the balance of the model, especially if class distributions are imbalanced.

### B. Confusion Matrix and Visual Heatmap
A mathematical confusion matrix is compiled and then visualized using a `seaborn` heatmap. 
- This visualization plots the **Actual Categories against the Predicted Categories**, providing an immediate graphical understanding of the model's true performance. 
- It effortlessly highlights True Positives, True Negatives, False Positives (Type I errors), and False Negatives (Type II errors), which are crucial to examine closely in medical imaging contexts.

### C. Precision-Recall Curve
Finally, the script calculates and plots a **Precision-Recall Curve**.
- This curve is a highly effective way to visualize the trade-off that occurs between Precision and Recall. 
- Plotting this helps the analyzer understand the model's predictive power across varying confidence thresholds and ensures the model is performing well across the board rather than just at a single default cut-off point.
