# Pneumonia Detection

## 🩺 Project Overview

This project is a deep learning–based image classification system designed to detect **Pneumonia** from chest X-ray images. It is built using the TensorFlow deep learning framework and wrapped in a beautiful, professional Flask-based web interface. Users can upload chest X-ray images and receive real-time predictions regarding the presence of pneumonia.

Pneumonia is a **serious respiratory infection** that causes **inflammation in the alveoli (air sacs)** of the lungs, often leading them to fill with fluid or pus. It can be caused by **bacteria**, **viruses**, or **fungi**, with **Streptococcus pneumoniae** being the most common bacterial cause. Pneumonia primarily affects **infants**, **elderly individuals**, and people with weakened immune systems, and if not diagnosed and treated early, it can result in severe complications or even death.

Traditionally, pneumonia is diagnosed using chest X-rays interpreted by radiologists. However, manual interpretation is time-consuming, error-prone, and may not be readily available in resource-constrained settings. This is where **Artificial Intelligence (AI)** and **Deep Learning (DL)** come into play.

Deep learning, particularly **Convolutional Neural Networks (CNNs)**, has shown immense success in processing medical images. By training on large datasets of chest X-ray images, CNNs can learn to automatically detect patterns associated with pneumonia — such as lung opacity, consolidation, or pleural effusion — which are often subtle and difficult to spot with the naked eye.

In this project, we utilize **Transfer Learning**, specifically **Xception** model (a high-performing CNN architecture), to classify X-ray images into two categories:
- **Normal**
- **Pneumonia**

This automated tool not only improves diagnostic speed and accuracy but also assists healthcare professionals by acting as a **second-opinion system**, thereby reducing workload and increasing confidence in clinical settings. Furthermore, by deploying the model as a Flask web application, it becomes accessible for both local and remote usage, making it a useful tool in **telemedicine** and **mobile healthcare solutions**.


## 📊 Performance Metrics
The model was evaluated using standard metrics:
- **Accuracy**
- **Loss**
- **Validation Accuracy**
- **Validation Loss**

We also used **ROC-AUC** to evaluate and determine the optimal classification threshold.

## 🗃️ Dataset
The dataset used is a labeled collection of **chest X-ray images** divided into:
- `Normal`
- `Pneumonia`

## 🛠️ How to Run
```bash
# Step 1: Clone the repository
$ git clone https://github.com/yourusername/pneumonia-detector.git

# Step 2: Navigate into project folder
$ cd pneumonia-detector

# Step 3: Install dependencies
$ pip install -r requirements.txt

# Step 4: Run the Flask app
$ python app.py

```

## 📌 Requirements
- Python 3.7+
- TensorFlow 2.x
- Flask
- OpenCV
- Matplotlib
- scikit-learn
- seaborn

Install them using:
```bash
pip install -r requirements.txt
```

## 📄 License
This project is licensed under the MIT License.
