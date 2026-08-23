# 🌱 Plant Growth Stage Detection

A Deep Learning-based web application that detects the **growth stage of a plant from an uploaded plant image**.

The project uses **MobileNetV2 Transfer Learning** to classify plant images into five different growth stages. A **Flask backend** handles image preprocessing and prediction, while a simple **HTML, CSS, and JavaScript frontend** allows users to upload a plant image and receive the predicted growth stage.

## 🔗 Project Links

**Live Application:**  
https://plant-growth-stage-detection.onrender.com

**GitHub Repository:**  
https://github.com/spriyag006/Plant-Growth-Stage-Detection-/tree/main

---

## 📌 Project Overview

Plant growth monitoring is important in agriculture because different growth stages require different care, nutrients, and environmental conditions.

This project uses Deep Learning and Computer Vision to automatically identify the growth stage of a plant from an image.

The user simply:

1. Uploads a plant image.
2. Clicks the **Predict** button.
3. The image is sent to the Flask backend.
4. The trained AI model analyzes the image.
5. The predicted growth stage and confidence score are displayed.

---

## 🎯 Objectives

- Detect plant growth stages automatically.
- Build an AI-based image classification system.
- Use Deep Learning for plant image classification.
- Implement Transfer Learning using MobileNetV2.
- Build a Flask REST API for prediction.
- Create a simple single-page frontend.
- Store prediction information using MongoDB Atlas.
- Deploy the backend using Render.

---

## 🧠 AI Model

The project uses **MobileNetV2 with Transfer Learning**.

MobileNetV2 is a lightweight Convolutional Neural Network (CNN) architecture designed for image classification.

Instead of training a CNN completely from scratch, the project uses a pretrained MobileNetV2 network for feature extraction and adds custom classification layers.

---

## Model Architecture


Plant Image
     │
     ▼
Image Preprocessing
     │
     ▼
Resize 224×224
     │
     ▼
MobileNetV2
     │
     ▼
Feature Extraction
     │
     ▼
Global Average Pooling
     │
     ▼
Dense Layer
     │
     ▼
Dropout
     │
     ▼
Softmax Classifier
     │
     ▼
Plant Growth Stage

---

## ▶️ Run the Backend

cd backend
python app.py
http://127.0.0.1:5000

---

## ☁️ Deployment

Root Directory: backend
Build Command:pip install -r ../requirements.txt
Start Command:gunicorn app:app
Python Version:3.12.10
TensorFlow Version:2.21.0

---

## 📄 License

This project is developed as a Deep Learning mini project for educational purposes.
---


## 👩‍💻 Author

Sangeetha Priya

Artificial Intelligence and Data Science

---
