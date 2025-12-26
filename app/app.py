
# import os
# import sqlite3
# from flask import Flask, render_template, request
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from datetime import datetime
# import cv2

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_DIR = os.path.join(BASE_DIR, "static/uploads")
# HEATMAP_DIR = os.path.join(BASE_DIR, "static/heatmaps")
# DB_PATH = os.path.join(BASE_DIR, "predictions.db")
# MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "model/bugvision_model.h5")

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(HEATMAP_DIR, exist_ok=True)

# # Load model
# model = tf.keras.models.load_model(MODEL_PATH)
# CLASSES = [
#     "UI_Error", 
#     "Database_Error", 
#     "Network_Error",
#     "Rendering_Error",
#     "Crash_Error",
#     "Other_Error"
# ]

# app = Flask(
#     __name__, 
#     template_folder=os.path.join(BASE_DIR, "templates"),
#     static_folder=os.path.join(BASE_DIR, "static")
# )

# # ============================================================
# # DB FUNCTIONS
# # ============================================================

# def db_connection():
#     return sqlite3.connect(DB_PATH)


# def log_prediction(image_path, heatmap_path, predicted_class, confidence):
#     conn = db_connection()
#     cursor = conn.cursor()
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     cursor.execute(
#         "INSERT INTO prediction_log (image_path, heatmap_path, predicted_class, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
#         (image_path, heatmap_path, predicted_class, confidence, ts)
#     )

#     conn.commit()
#     conn.close()


# def get_recent_predictions(limit=10):
#     conn = db_connection()
#     cursor = conn.cursor()

#     cursor.execute(
#         "SELECT image_path, heatmap_path, predicted_class, confidence, timestamp FROM prediction_log ORDER BY id DESC LIMIT ?",
#         (limit,)
#     )
#     rows = cursor.fetchall()
#     conn.close()

#     return rows


# def get_class_counts():
#     conn = db_connection()
#     cursor = conn.cursor()

#     cursor.execute(
#         "SELECT predicted_class, COUNT(*) FROM prediction_log GROUP BY predicted_class"
#     )
#     rows = cursor.fetchall()
#     conn.close()

#     return rows

# # ============================================================
# # Grad-CAM Heatmap
# # ============================================================

# def make_gradcam_overlay(model, img_path, class_index=None):
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
#     img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     last_conv = model.get_layer("Conv_1")
#     grad_model = tf.keras.Model([model.inputs], [last_conv.output, model.output])

#     with tf.GradientTape() as tape:
#         conv_out, preds = grad_model(img_array)
#         if class_index is None:
#             class_index = np.argmax(preds[0])
#         loss = preds[:, class_index]

#     grads = tape.gradient(loss, conv_out)[0]
#     pooled_grads = np.mean(grads, axis=(0, 1))
#     conv_out = conv_out[0]

#     for i in range(pooled_grads.shape[-1]):
#         conv_out[:, :, i] *= pooled_grads[i]

#     heatmap = np.mean(conv_out, axis=-1)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= heatmap.max()

#     original = cv2.imread(img_path)
#     original = cv2.resize(original, (256, 256))

#     heat = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
#     overlay = cv2.addWeighted(original, 0.65, heat, 0.45, 0)

#     out_path = os.path.join(HEATMAP_DIR, "heatmap_" + os.path.basename(img_path))
#     cv2.imwrite(out_path, overlay)

#     return "/static/heatmaps/" + os.path.basename(out_path)

# # ============================================================
# # Preprocessing
# # ============================================================

# def preprocess(path):
#     img = Image.open(path).convert("RGB")
#     img = img.resize((256, 256))
#     arr = np.array(img) / 255.0
#     return np.expand_dims(arr, axis=0)

# # ============================================================
# # ROUTES
# # ============================================================

# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     file = request.files["file"]

#     filename = datetime.now().strftime("%Y%m%d%H%M%S_") + file.filename
#     file_path = os.path.join(UPLOAD_DIR, filename)
#     file.save(file_path)

#     img = preprocess(file_path)
#     probs = model.predict(img)[0]
#     class_idx = np.argmax(probs)

#     predicted = CLASSES[class_idx]
#     confidence = round(float(probs[class_idx] * 100), 2)

#     heatmap_url = make_gradcam_overlay(model, file_path, class_idx)

#     log_prediction(
#         image_path="/static/uploads/" + filename,
#         heatmap_path=heatmap_url,
#         predicted_class=predicted,
#         confidence=confidence
#     )

#     return render_template(
#         "result.html",
#         image="/static/uploads/" + filename,
#         heatmap=heatmap_url,
#         predicted_class=predicted,
#         confidence=confidence,
#         classes=CLASSES,
#         all_probs=[round(float(p) * 100, 2) for p in probs]
#     )


# @app.route("/dashboard")
# def dashboard():
#     return render_template(
#         "dashboard.html",
#         recent=get_recent_predictions(),
#         class_counts=get_class_counts()
#     )


# @app.route("/logs")
# def logs():
#     conn = db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM prediction_log ORDER BY id DESC")
#     rows = cursor.fetchall()
#     conn.close()

#     return render_template("logs.html", logs=rows)


# # ============================================================
# # RUN FLASK SERVER
# # ============================================================
# if __name__ == "__main__":
#     app.run(debug=False)
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import os
import sqlite3
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime

import cv2



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static/uploads")
HEATMAP_DIR = os.path.join(BASE_DIR, "static/heatmaps")
DB_PATH = os.path.join(BASE_DIR, "predictions.db")
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "model/bugvision_model.h5")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)




model = tf.keras.models.load_model(MODEL_PATH, compile=False)



CLASSES = [
    "UI_Error",
    "Database_Error",
    "Network_Error",
    "Rendering_Error",
    "Crash_Error",
    "Other_Error"
]



app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)


def db_connection():
    return sqlite3.connect(DB_PATH)

def log_prediction(image_path, heatmap_path, predicted_class, confidence):
    conn = db_connection()
    cursor = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO prediction_log (image_path, heatmap_path, predicted_class, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (image_path, heatmap_path, predicted_class, confidence, ts))

    conn.commit()
    conn.close()

def get_recent_predictions(limit=10):
    conn = db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT image_path, heatmap_path, predicted_class, confidence, timestamp
        FROM prediction_log ORDER BY id DESC LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()
    return rows

def get_class_counts():
    conn = db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT predicted_class, COUNT(*)
        FROM prediction_log GROUP BY predicted_class
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def make_gradcam_overlay(model, img_path, class_index=None):

    # Load + preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Detect correct last conv layer
    layer_names = [layer.name for layer in model.layers]
    last_conv_name = "Conv_1" if "Conv_1" in layer_names else "out_relu"
    last_conv_layer = model.get_layer(last_conv_name)

    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(img_array)

        # Fix prediction shape issue
        if isinstance(predictions, list):
            predictions = predictions[0]

        if class_index is None:
            class_index = tf.argmax(predictions[0])

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_out)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_out = conv_out[0] * pooled_grads

    heatmap = np.mean(conv_out, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    original = cv2.imread(img_path)

    original = cv2.resize(original, (256, 256))


    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * cv2.resize(heatmap, (256, 256))), 
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(original, 0.65, heatmap_color, 0.45, 0)

    out_path = os.path.join(HEATMAP_DIR, "heatmap_" + os.path.basename(img_path))
    cv2.imwrite(out_path, overlay)

    return "/static/heatmaps/" + os.path.basename(out_path)


def preprocess(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    filename = datetime.now().strftime("%Y%m%d%H%M%S_") + file.filename
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    img = preprocess(file_path)
    probs = model.predict(img)[0]
    class_idx = np.argmax(probs)

    predicted = CLASSES[class_idx]
    confidence = round(float(probs[class_idx] * 100), 2)

    heatmap_url = make_gradcam_overlay(model, file_path, class_idx)

    log_prediction(
        image_path="/static/uploads/" + filename,
        heatmap_path=heatmap_url,
        predicted_class=predicted,
        confidence=confidence
    )

 

    probs_list = [round(float(p) * 100, 2) for p in probs]
    combined = list(zip(CLASSES, probs_list))

    return render_template(
    "result.html",
    image="/static/uploads/" + filename,
    heatmap=heatmap_url,
    predicted_class=predicted,
    confidence=confidence,
    combined=combined
)


@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        recent=get_recent_predictions(),
        class_counts=get_class_counts()
    )

@app.route("/logs")
def logs():
    conn = db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_log ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    return render_template("logs.html", logs=rows)




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
