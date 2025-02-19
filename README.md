# Melanoma Skin Cancer Detection

## Abstract

In the realm of cancer, there exist over 200 distinct forms, with melanoma standing out as the most lethal type of skin cancer among them. The diagnostic protocol for melanoma typically initiates with clinical screening, followed by dermoscopic analysis and histopathological examination. Early detection of melanoma skin cancer is pivotal, as it significantly enhances the chances of successful treatment. The initial step in diagnosing melanoma skin cancer involves visually inspecting the affected area of the skin. Dermatologists capture dermatoscopic images of the skin lesions using high-speed cameras, which yield diagnostic accuracies ranging from 65% to 80% for melanoma without supplementary technical assistance. Through further visual assessment by oncologists and dermatoscopic image analysis, the overall predictive accuracy of melanoma diagnosis can be elevated to 75% to 84%. The objective of the project is to construct an automated classification system leveraging image processing techniques to classify skin cancer based on images of skin lesions.

## Problem statement

To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

## Table of Contents

- [General Info](#general-information)
- [Model Architecture](#model-architecture)
- [Model Summary](#model-summary)
- [Model Evaluation](#model-evaluation)
- [Technologies Used](#technologies-used)
- [Acknowledgements](#acknowledgements)
- [Collaborators](#collaborators)

<!-- You can include any other section that is pertinent to your problem -->

## General Information

The dataset comprises 2357 images depicting malignant and benign oncological conditions, sourced from the International Skin Imaging Collaboration (ISIC). These images were categorized based on the classification provided by ISIC, with each subset containing an equal number of images.

In order to address the challenge of class imbalance, the Augmentor Python package (https://augmentor.readthedocs.io/en/master/) was employed to augment the dataset. This involved generating additional samples for all classes, ensuring that none of the classes had insufficient representation.

## Pictorial representation of skin types

The aim of this task is to assign a specific class label to a particular type of skin cancer.

## Model Architecture

### **Summary of the CNN Model Architecture**  

This convolutional neural network (CNN) is built using the **Keras Sequential API** for a **9-class classification problem**. The architecture includes the following layers:

1. **Data Augmentation Layer**  
   - Applies transformations to enhance dataset variability.

2. **Rescaling Layer**  
   - Normalizes pixel values to the \([0,1]\) range.

3. **Convolutional & Pooling Layers**  
   - **4 Convolutional layers** with increasing filters: **32, 64, 128, and 256**.  
   - Each Conv2D layer uses a **3×3 kernel** and **ReLU activation**.  
   - **3 MaxPooling layers** (2×2) for downsampling.  
   - **Batch Normalization** after the first convolution block to stabilize training.  

4. **Regularization (Dropout)**  
   - A **0.25 dropout** after the last Conv2D layer.  
   - Additional **dropout layers (0.3, 0.2)** between fully connected layers to reduce overfitting.

5. **Fully Connected Layers (Dense Layers)**  
   - **Flattening layer** to convert feature maps into a vector.  
   - **Dense layers (256 → 128)** with ReLU activation for feature learning.  
   - **Final Dense layer (9 units, softmax activation)** for multi-class classification.

This model is structured to balance feature extraction, dimensionality reduction, and overfitting prevention, making it well-suited for image classification tasks. 🚀

## Model Summar

**Summary:**

These graphs depict the training and validation performance of a machine learning model over 50 epochs.

* **Accuracy (Left Graph):**
    * The training accuracy steadily increases over epochs, reaching nearly 90% by the end.
    * The validation accuracy also increases, but with more fluctuations, and plateaus around 85-87%.
    * The gap between training and validation accuracy suggests some degree of overfitting, as the model performs slightly better on the training data than on unseen validation data.

* **Loss (Right Graph):**
    * The training loss rapidly decreases in the early epochs and continues to decline, albeit at a slower rate, reaching a low value by epoch 50.
    * The validation loss also decreases initially, but shows more variability and plateaus, indicating the model's performance on unseen data is no longer improving significantly.
    * The Loss graph further confirms the possibility of overfitting.

**In essence, the model is learning well on the training data but is starting to show signs of overfitting. Further steps like regularization or early stopping might be needed to improve generalization.**

## Model Evaluation
Absolutely, let's break down a model evaluation based on the graphs provided:

**Model Evaluation:**


**1. Training Performance:**

* **Accuracy:** The model demonstrates good learning on the training data, achieving a high accuracy of nearly 90% by the end of training. This indicates the model is capable of fitting the training patterns effectively.
* **Loss:** The training loss decreases significantly and consistently, confirming that the model is learning to minimize the error on the training data.

**2. Validation Performance:**

* **Accuracy:** The validation accuracy also improves, suggesting the model is generalizing to unseen data. However, it plateaus around 85-87%, indicating that the model's ability to generalize has reached a limit.
* **Loss:** The validation loss decreases initially, but shows fluctuations and plateaus, suggesting that further training is not significantly improving the model's performance on the validation set.

**3. Overfitting:**

* **Gap between Training and Validation Accuracy:** The noticeable gap between training and validation accuracy suggests overfitting. The model is performing better on the training data than on the validation data, indicating it might be memorizing training patterns rather than learning generalizable features.
* **Plateauing Validation Loss:** The plateauing validation loss, despite continued decrease in training loss, is another indicator of overfitting.

**4. Model Stability and Convergence:**

* **Fluctuations in Validation Metrics:** The fluctuations in the validation accuracy and loss curves suggest that the model's performance on the validation set is not entirely stable. This could be due to factors like the size of the validation set or the complexity of the model.
* **Convergence:** The training loss appears to have converged, but the validation metrics suggest that the model has reached a point of diminishing returns in terms of generalization.

**5. Potential Improvements:**

* **Regularization:** Techniques like L1 or L2 regularization can help reduce overfitting by penalizing complex models.
* **Early Stopping:** Monitoring the validation loss and stopping training when it starts to increase can prevent overfitting and save computational resources.
* **Data Augmentation:** Increasing the size and diversity of the training data can help the model learn more robust features and improve generalization.
* **Hyperparameter Tuning:** Experimenting with different hyperparameters, such as learning rate, batch size, and network architecture, can potentially lead to better performance.

**Conclusion:**

The model shows promising results on the training data, but exhibits signs of overfitting. Further steps are needed to improve generalization and achieve better performance on unseen data. The validation metrics suggest that the model has reached a point where further training without adjustments is unlikely to yield significant improvements. The suggested improvements focus on mitigating overfitting and enhancing the model's ability to generalize.


## Technologies Used

- [Python](https://www.python.org/) - version 3.11.4
- [Matplotlib](https://matplotlib.org/) - version 3.7.1
- [Numpy](https://numpy.org/) - version 1.24.3
- [Pandas](https://pandas.pydata.org/) - version 1.5.3
- [Seaborn](https://seaborn.pydata.org/) - version 0.12.2
- [Tensorflow](https://www.tensorflow.org/) - version 2.15.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements

- UpGrad tutorials on Convolution Neural Networks (CNNs) on the learning platform

- [Melanoma Skin Cancer](https://www.cancer.org/cancer/melanoma-skin-cancer/about/what-is-melanoma.html)

- [Introduction to CNN](https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/)

- [Image classification using CNN](https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/)

- [Efficient way to build CNN architecture](https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7)

## Collaborators

Created by [@sowmily](https://github.com/sowmilydutta)
