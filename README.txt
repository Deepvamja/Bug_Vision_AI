
BugVision AI — Deep Learning Bug Screenshot Classifier
======================================================

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
