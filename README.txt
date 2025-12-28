🐞 BugVision AI
Intelligent Bug Screenshot Classification & Explainability System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN-green)
![Explainable AI](https://img.shields.io/badge/Explainable%20AI-Grad--CAM-purple)

BugVision AI is a production-style deep learning system that automatically classifies software bug screenshots into real-world error categories and explains why the model made each decision using visual heatmaps.

🚨 The Problem (Real & Industrial)

In real software teams:

Bug reports arrive as screenshots

Engineers manually inspect & triage them

High volume leads to slow debugging and misclassification

No visibility into recurring bug patterns

❌ Manual triage
❌ Delayed fixes
❌ Inconsistent categorization

💡 The Solution

BugVision AI automates bug triaging using computer vision.

It:

Accepts bug screenshots

Classifies them into 6 real-world error types

Assigns confidence scores

Generates Grad-CAM heatmaps to explain predictions

Logs predictions and displays analytics

Runs as a deployed web application

This is not a notebook demo — it’s a complete system.

🏷️ Supported Bug Categories

UI Error – layout, buttons, UI breakage

Database Error – SQL, DB connection failures

Network Error – timeouts, connectivity issues

Rendering Error – blank screens, visual glitches

Crash Error – fatal app crashes

Other Error – rare or mixed cases

✨ Key Features (What Makes It Stand Out)

CNN-based image classification

Explainable AI (Grad-CAM) — not a black box

End-to-end ML inference pipeline

Flask-based production web app

Prediction logging & analytics dashboard

Cloud deployment with real runtime constraints

Most student projects stop at training.
This project goes all the way to deployment and failure handling.

🧠 System Flow
Screenshot Upload
        ↓
Image Preprocessing (256×256)
        ↓
CNN Model Inference
        ↓
Softmax Probabilities
        ↓
Grad-CAM Explainability
        ↓
Flask App + Dashboard

🛠️ Tech Stack

ML / DL: TensorFlow, CNN, Grad-CAM
Backend: Python, Flask
Computer Vision: OpenCV, NumPy
Data: SQLite



HOW TO RUN LOCALLY
------------------
1. Create virtual environment (optional but recommended)
   python -m venv venv

2. Activate environment
   (Windows) venv\Scripts\activate
   (Mac/Linux) source venv/bin/activate

3. Install dependencies
   pip install -r requirements.txt

4. Run Flask server
   python app/app.py

5. Open in browser:
   http://127.0.0.1:5000


PROJECT STRUCTURE
-----------------
- app.py (backend)
- templates/ (HTML files)
- static/uploads (uploaded screenshots)
- static/heatmaps (Grad-CAM heatmaps)
- static/charts (training graphs)
- model/bugvision_model.h5 (trained deep learning model)
- predictions.db (prediction log database)
