<h1 align="center">🩺 Intelligent Thyroid Diagnosis System</h1>

---

### 🚀 Flask • Machine Learning • Deep Learning (YOLOv8) • SQLite Authentication

This is a medical diagnostic web application that predicts thyroid disorders using machine learning and detects thyroid cancer cells using YOLOv8 object detection.  
The system includes user authentication (OTP-based signup), symptoms-based prediction, value-based prediction, and image-based cancer analysis.

---
A complete medical diagnostic system that performs:

🔹 **Thyroid Disease Prediction** (Numeric Input)  
🔹 **Thyroid Disease Prediction** (Symptoms-Based ML + Neural Network)  
🔹 **Thyroid Cancer Detection** (YOLOv8 Object Detection)    

This project combines **Machine Learning**, **Deep Learning**, and **Computer Vision** into one unified health-diagnosis platform.

---

## 🧬 Features Overview

### 🔥 1. Thyroid Disease Prediction (Numeric Values)
Predicts:
- ✅ Normal  
- ⚠️ Thyroid Disorder (Sick)

 Based on medical inputs like **TSH, T3, T4**, etc.  
 The backend loads a classical ML model for this prediction.

 <img src="static/images/Screenshot 2025-11-24 153430.png" width="70%" alt="Project Logo">

### 🧠 2. Symptoms-Based Thyroid Prediction (Advanced Pipeline)
Powered by:
- 🔹 Categorical Encoding  
- 🔹 Neural Network Feature Extraction  
- 🔹 Ensemble Classifier(Voting/Bagging/Stacking)
- 🔹 Meta-Classifier (Final Prediction Stage)

<img src="static/images/Screenshot 2025-11-24 151018.png" width="70%" alt="Project Logo">


Outputs:
- 🎯 Disease Name  
- 📘 Disease Description  
- 🛡️ Precautionary Measures  
- 🥗 Food & Diet Recommendations  

<img src="static/images/Screenshot 2025-11-24 150948.png" width="70%" alt="Project Logo">

### 🩻 3. Thyroid Cancer Detection (YOLOv8)
Upload a thyroid cell/tissue image → system detects:

- **TC**
- **Normal**
- **Abnormal**
- **Inflammatory cells**
- **Stroma**
- **RBC**
- **Fibromuscular tissue**
- **Lymphoid cells**

Output includes:
- Bounding boxes  
- Confidence scores  
- Annotated image generated & saved  
- JSON response for API usage  

<img src="static/images/Screenshot 2025-11-24 151110.png" width="70%" alt="Project Logo">

---

## Technology Stack

## 🛠️ Tech Stack

- **Backend:** Flask, Python  
- **ML / DL:** TensorFlow, Scikit-Learn, Joblib  
- **Object Detection:** YOLOv8 (Ultralytics)  
- **Database:** SQLite  
- **Frontend:** HTML, CSS, JS  
- **Image Processing:** OpenCV  

---

## Installation

1. Install Dependencies
```
pip install flask numpy pandas scikit-learn tensorflow joblib ultralytics opencv-python flask-cors
```
2. Run the Flask Server

```
python app.py
```
4. Access the App

```
http://127.0.0.1:5000/
```
---

## 🔍 How It Works (Backend Flow)
### 🔹 Numeric Prediction  
1.Accept user inputs  
2.Convert to array  
3.ML model → Predict Normal / Sick

### ✔ Symptom-Based Prediction  
1.Encode categorical features  
2.Scale numerical features  
3.Extract deep features via NN  
4.Hybrid predictions (Voting/Bagging/Stacking)  
5.Combined meta-classifier output  
6.Display disease + description + precautions + food plan  

### ✔ Cancer Detection  
1.Validate uploaded image  
2.Run inference using YOLOv8   
3.Draw bounding boxes and labels  
4.Save annotated result in /static/processed  
5.Return JSON response  
