🐞 BugVision AI
Intelligent Bug Screenshot Classification & Explainability System

BugVision AI is a production-style deep learning system that automatically classifies software bug screenshots into real-world error categories and explains why the model made each decision using visual heatmaps.

🚀 The Problem (Real & Industrial):

In modern software teams:

QA engineers report bugs using screenshots

Developers manually read and triage bugs

Large teams receive hundreds of screenshots daily

Bug categorization becomes slow, error-prone, and inconsistent

❌ Manual triaging
❌ Delayed debugging
❌ No visibility into recurring bug patterns

BugVision AI solves this by automating bug classification using computer vision.


💡 The Solution

🧠 What BugVision AI Does

✔️ Accepts bug screenshots as input

✔️ Classifies them into 6 real-world error types

✔️ Generates confidence scores

✔️ Explains predictions using Grad-CAM heatmaps

✔️ Stores prediction history

✔️ Provides an analytics dashboard

✔️ Runs as a full-stack deployed application



🏷️ Supported Bug Categories

Class	        Description

UI_Error	Layout issues, misaligned buttons, broken UI elements

Database_Error	SQL errors, DB connection failures

Network_Error	Timeouts, connectivity issues

Rendering_Error	Graphics glitches, blank screens

Crash_Error	App crashes, fatal error screens

Other_Error	Unclassified or rare error types


✨ Key Features

🧠 Automated Bug Triage

Automatically categorizes bug screenshots into real-world error types, eliminating manual inspection and reducing triage time.

🔍 Explainable AI for Trust & Debugging

Generates Grad-CAM heatmaps that visually explain model decisions, helping engineers verify predictions instead of blindly trusting AI.

⚙️ Production-Ready Inference Pipeline

Implements a complete ML inference workflow including preprocessing, prediction, confidence scoring, and result rendering — mirroring real production systems.

📊 Engineering Analytics Dashboard

Tracks prediction history and class distributions, enabling teams to identify recurring bug patterns and systemic issues.

🗃️ Persistent Prediction Logging

Stores inference results with timestamps, supporting auditing, debugging, and future model improvements.


🧩 System Architecture

User Screenshot
      ↓
Image Preprocessing (256×256 RGB)
      ↓
CNN-based Deep Learning Model
      ↓
Softmax Probability Distribution
      ↓
Grad-CAM Explainability
      ↓
Flask Web Application
      ↓
Prediction Logging + Dashboard



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


