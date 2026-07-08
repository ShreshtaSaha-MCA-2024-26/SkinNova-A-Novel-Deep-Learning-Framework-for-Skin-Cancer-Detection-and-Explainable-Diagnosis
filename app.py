# ==========================================================
# SkinNova - AI Powered Skin Cancer Detection System
# Part 1 : Configuration, Model Loading & Prediction Engine
# ==========================================================

import os
import logging
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)

from werkzeug.utils import secure_filename
from pymongo import MongoClient
from datetime import datetime

# ==========================================================
# Flask Application
# ==========================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = "SkinNova_Secret_Key_2026"

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

MODEL_PATH = BASE_DIR / "saved_model" / "best_model.keras"

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("SkinNova")

# ==========================================================
# MongoDB Connection
# ==========================================================

try:
    client = MongoClient("mongodb://127.0.0.1:27017/")

    db = client["skinnova"]

    prediction_collection = db["predictions"]

    logger.info("MongoDB Connected Successfully")

except Exception as e:

    logger.exception("MongoDB Connection Failed")

    raise RuntimeError(f"MongoDB Error: {e}")
# ==========================================================
# Model Configuration
# ==========================================================

IMG_SIZE = 224

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

CLASS_NAMES = {
    0: "Benign",
    1: "Malignant"
}


# ==========================================================
# Load Trained Model
# ==========================================================

logger.info("=" * 60)
logger.info("Loading AI Model...")
logger.info("=" * 60)

try:

    model = tf.keras.models.load_model(MODEL_PATH)

    logger.info("Model Loaded Successfully")

except Exception as e:

    logger.exception("Unable to load model")

    raise RuntimeError(
        f"Cannot load model : {e}"
    )


# ==========================================================
# Find Last Convolution Layer Automatically
# ==========================================================

LAST_CONV_LAYER = None

for layer in reversed(model.layers):

    try:

        if len(layer.output_shape) == 4:

            LAST_CONV_LAYER = layer.name

            break

    except Exception:

        continue

logger.info(f"Last Conv Layer : {LAST_CONV_LAYER}")


# ==========================================================
# Helper Functions
# ==========================================================

def allowed_file(filename: str) -> bool:
    """
    Check whether uploaded file extension is allowed.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess_image(image_path):
    """
    Reads an image and converts it into
    model input tensor.
    """

    try:
        image = Image.open(image_path)
        image.verify()          # Verify file integrity

        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Invalid image file: {e}")


    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = np.asarray(image).astype(np.float32)

    # ------------------------------------------------------
    # IMPORTANT
    #
    # Uncomment ONLY if your model was trained with:
    #
    # image = image / 255.0
    #
    # ------------------------------------------------------

    # image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image


# ==========================================================
# Prediction Function
# ==========================================================

def predict_image(image_path):
    """
    Predict whether the lesion is
    Benign or Malignant.
    """

    image = preprocess_image(image_path)

    probability = float(
        model.predict(
            image,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:

        prediction_index = 1

        confidence = probability

        risk = "High Risk"

    else:

        prediction_index = 0

        confidence = 1 - probability

        risk = "Low Risk"

    return {

        "prediction": CLASS_NAMES[prediction_index],

        "confidence": round(
            confidence * 100,
            2
        ),

        "probability": round(
            probability,
            4
        ),

        "risk": risk
    }


# ==========================================================
# Store Latest Prediction
# ==========================================================

latest_result = {}


logger.info("=" * 60)
logger.info("SkinNova Initialization Completed")
logger.info("=" * 60)


# ==========================================================
# Part 2 : Grad-CAM, File Upload & Prediction Wrapper
# ==========================================================

def make_gradcam_heatmap(img_array):
    """
    Generate Grad-CAM heatmap for the uploaded image.
    """

    try:

        # Backbone (EfficientNetB0)
        base_model = model.get_layer("efficientnetb0")

        last_conv_layer = base_model.get_layer("top_conv")

        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=last_conv_layer.output
        )

        classifier_input = tf.keras.Input(
            shape=last_conv_layer.output.shape[1:]
        )

        x = classifier_input

        start = False

        for layer in model.layers:

            if layer.name == "global_average_pooling2d":

                start = True

            if start:

                x = layer(x)

        classifier_model = tf.keras.models.Model(
            classifier_input,
            x
        )

        with tf.GradientTape() as tape:

            conv_outputs = grad_model(img_array)

            tape.watch(conv_outputs)

            predictions = classifier_model(conv_outputs)

            loss = predictions[:, 0]

        gradients = tape.gradient(
            loss,
            conv_outputs
        )

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]

        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)

        heatmap /= (
            tf.reduce_max(heatmap) + 1e-10
        )

        return heatmap.numpy()

    except Exception as e:

        logger.exception(e)

        raise RuntimeError(
            f"GradCAM Error : {e}"
        )


# ==========================================================
# Save GradCAM Overlay
# ==========================================================

def save_gradcam(image_path, heatmap, save_path):
    """
    Save Grad-CAM heatmap overlay image.
    """

    image = cv2.imread(str(image_path))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        0.60,
        heatmap,
        0.40,
        0
    )

    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(save_path),
        overlay
    )


# ==========================================================
# Prediction + GradCAM
# ==========================================================

def predict_with_heatmap(image_path):
    """
    Perform prediction and generate Grad-CAM.
    """

    image = preprocess_image(image_path)

    probability = float(
        model.predict(
            image,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:

        prediction = "Malignant"

        confidence = probability

        risk = "High Risk"

    else:

        prediction = "Benign"

        confidence = 1 - probability

        risk = "Low Risk"

    heatmap = make_gradcam_heatmap(image)

    filename = Path(image_path).name

    heatmap_filename = f"heatmap_{filename}"

    heatmap_path = UPLOAD_FOLDER / heatmap_filename

    save_gradcam(
        image_path,
        heatmap,
        heatmap_path
    )

    return {

        "prediction": prediction,

        "confidence": round(
            confidence * 100,
            2
        ),

        "probability": round(
            probability,
            4
        ),

        "risk": risk,

        "image": filename,

        "heatmap": heatmap_filename
    }


# ==========================================================
# Save Uploaded Image
# ==========================================================

def save_uploaded_file(file):
    """
    Save uploaded image securely.
    """

    filename = secure_filename(file.filename)

    filepath = UPLOAD_FOLDER / filename

    file.save(filepath)

    logger.info(f"Uploaded : {filename} ({os.path.getsize(filepath)} bytes)")

    return filename, filepath


# ==========================================================
# Run Complete Prediction Pipeline
# ==========================================================

def run_prediction(file):
    """
    Complete prediction workflow.

    IMPORTANT: This function reads and saves the uploaded
    file's stream. It must be called ONLY ONCE per request,
    since the underlying file stream is exhausted after the
    first .save() call — calling this twice on the same
    `file` object will save an empty (0-byte) file the
    second time around.
    """

    global latest_result

    filename, filepath = save_uploaded_file(file)

    latest_result = predict_with_heatmap(filepath)

    logger.info(
        f"Prediction : {latest_result['prediction']}"
    )

    return latest_result


# ==========================================================
# Utility Function
# ==========================================================

def clear_latest_result():
    """
    Clear cached prediction result.
    """

    global latest_result

    latest_result = {}



# ==========================================================
# Part 3 : Flask Routes
# ==========================================================


# -------------------------------
# Home Page
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# About Page
# -------------------------------
@app.route("/about")
def about():
    return render_template("about.html")


# -------------------------------
# Dashboard
# -------------------------------
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# -------------------------------
# Research
# -------------------------------
@app.route("/research")
def research():
    return render_template("research.html")


# -------------------------------
# Skin Cancer Information
# -------------------------------
@app.route("/skin-cancer-info")
def skin_cancer_info():
    return render_template("skin_cancer_info.html")


# -------------------------------
# Explainability
# -------------------------------
@app.route("/explain")
def explain():

    if not latest_result:
        flash("Please perform a detection first.")
        return redirect(url_for("detect"))

    return render_template(
        "explain.html",
        result=latest_result
    )


# -------------------------------
# Recommendation
# -------------------------------
@app.route("/recommendation")
def recommendation():

    if not latest_result:
        flash("Please perform a detection first.")
        return redirect(url_for("detect"))

    return render_template(
        "recommendation.html",
        result=latest_result
    )


# -------------------------------
# Report
# -------------------------------
@app.route("/report")
def report():

    if not latest_result:
        flash("Please perform a detection first.")
        return redirect(url_for("detect"))

    return render_template(
        "report.html",
        result=latest_result
    )


# ==========================================================
# Detection Page
# ==========================================================

@app.route("/detect", methods=["GET", "POST"])
def detect():

    if request.method == "POST":

        # ----------------------------------------------
        # Validate the incoming file BEFORE running
        # prediction. run_prediction() must only be
        # called ONCE per uploaded file.
        # ----------------------------------------------

        if "image" not in request.files:

            return render_template(
                "detect.html",
                error="Please select an image."
            )

        file = request.files["image"]

        if file.filename == "":

            return render_template(
                "detect.html",
                error="Please select an image."
            )

        if not allowed_file(file.filename):

            logger.warning(
                f"Rejected file with disallowed extension: {file.filename}"
            )

            return render_template(
                "detect.html",
                error="Only JPG, JPEG and PNG images are allowed."
            )

        try:

            result = run_prediction(file)

            prediction_collection.insert_one({

                "prediction": result["prediction"],

                "confidence": result["confidence"],

                "probability": result["probability"],

                "risk": result["risk"],

                "image": result["image"],

                "heatmap": result["heatmap"],

                "model": "EfficientNetB0",

                "created_at": datetime.now()

            })

            return render_template(

                "detect.html",

                prediction=result["prediction"],

                confidence=result["confidence"],

                probability=result["probability"],

                risk=result["risk"],

                filename=result["image"],

                heatmap=result["heatmap"]

            )

        except Exception as e:

            logger.exception("Prediction pipeline failed")

            return render_template(
                "detect.html",
                error=str(e)
            )

    return render_template("detect.html")

# ==========================================================
# Result Page
# ==========================================================

@app.route("/result")
def result():

    if not latest_result:

        flash("Please upload an image first.")

        return redirect(url_for("detect"))

    return render_template(

        "result.html",

        prediction=latest_result["prediction"],

        confidence=latest_result["confidence"],

        probability=latest_result["probability"],

        risk=latest_result["risk"],

        image=latest_result["image"],

        heatmap=latest_result["heatmap"]

    )


# ==========================================================
# Reset Prediction
# ==========================================================

@app.route("/reset")
def reset():

    clear_latest_result()

    flash("Prediction cleared successfully.")

    return redirect(url_for("detect"))



# ==========================================================
# Part 4 : Error Handlers & Application Runner
# ==========================================================

# ----------------------------------------------------------
# 404 - Page Not Found
# ----------------------------------------------------------
@app.errorhandler(404)
def page_not_found(error):

    logger.warning(
        f"404 Error : {request.path}"
    )

    return render_template(
        "404.html"
    ), 404


# ----------------------------------------------------------
# 500 - Internal Server Error
# ----------------------------------------------------------
@app.errorhandler(500)
def internal_server_error(error):

    logger.exception(
        "Internal Server Error"
    )

    return render_template(
        "500.html"
    ), 500


# ----------------------------------------------------------
# Global Exception Logger
# ----------------------------------------------------------
# @app.errorhandler(Exception)
# def handle_exception(error):

#     logger.exception(error)

#     return render_template(
#         "500.html",
#         error=str(error)
#     ), 500


# ==========================================================
# Application Startup Banner
# ==========================================================

def startup_banner():

    print("\n")

    print("=" * 70)
    print("             SkinNova AI Skin Cancer Detection")
    print("=" * 70)

    print(f"Model Path      : {MODEL_PATH}")
    print(f"Upload Folder   : {UPLOAD_FOLDER}")
    print(f"Image Size      : {IMG_SIZE} x {IMG_SIZE}")
    print(f"Classes         : {list(CLASS_NAMES.values())}")
    print(f"Last Conv Layer : {LAST_CONV_LAYER}")

    print("=" * 70)
    print("Server Started Successfully")
    print("Open Browser : http://127.0.0.1:5000")
    print("=" * 70)
    print("\n")


# ==========================================================
# Run Application
# ==========================================================

if __name__ == "__main__":

    startup_banner()

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True,

        threaded=True

    )