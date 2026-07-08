"""
=========================================================
SkinNova
Train.py (Part 1A)

Author : Soumya Deep Saha
=========================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import cv2
import glob
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ==========================================================
# RANDOM SEED
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================================
# GPU CONFIGURATION
# ==========================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("GPU Enabled")

    except RuntimeError as e:
        print(e)

print("TensorFlow :", tf.__version__)

# ==========================================================
# MIXED PRECISION
# ==========================================================

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

print("Mixed Precision Enabled")

# ==========================================================
# PROJECT DIRECTORY
# ==========================================================

PROJECT_DIR = Path.cwd()

DATASET_DIR = PROJECT_DIR / "Dataset"

OUTPUT_DIR = PROJECT_DIR / "Outputs"

MODEL_DIR = OUTPUT_DIR / "models"

GRAPH_DIR = OUTPUT_DIR / "graphs"

CSV_DIR = OUTPUT_DIR / "csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# DATASET PATH
# ==========================================================

metadata_path = DATASET_DIR / "HAM10000_metadata.csv"

image_dir1 = DATASET_DIR / "HAM10000_images_part_1"

image_dir2 = DATASET_DIR / "HAM10000_images_part_2"

# ==========================================================
# LOAD METADATA
# ==========================================================

metadata = pd.read_csv(metadata_path)

print(metadata.head())

print()

print("Dataset Shape :", metadata.shape)

# ==========================================================
# IMAGE PATH MAPPING
# ==========================================================

image_paths = {}

for folder in [image_dir1, image_dir2]:

    for img in glob.glob(str(folder / "*.jpg")):

        image_id = os.path.basename(img).split(".")[0]

        image_paths[image_id] = img

metadata["image_path"] = metadata["image_id"].map(image_paths)

print("Missing Images :", metadata["image_path"].isnull().sum())

# ==========================================================
# REMOVE MISSING IMAGES
# ==========================================================

metadata = metadata.dropna(subset=["image_path"])

metadata.reset_index(drop=True, inplace=True)

# ==========================================================
# LABEL ENCODING
# ==========================================================

encoder = LabelEncoder()

metadata["label"] = encoder.fit_transform(metadata["dx"])

NUM_CLASSES = metadata["label"].nunique()

CLASS_NAMES = encoder.classes_

print()

print(CLASS_NAMES)

print()

print("Classes :", NUM_CLASSES)

# ==========================================================
# TRAIN VALID TEST SPLIT
# ==========================================================

train_df, temp_df = train_test_split(

    metadata,

    test_size=0.30,

    random_state=SEED,

    stratify=metadata["label"]

)

valid_df, test_df = train_test_split(

    temp_df,

    test_size=0.50,

    random_state=SEED,

    stratify=temp_df["label"]

)

print()

print("Training :", len(train_df))

print("Validation :", len(valid_df))

print("Testing :", len(test_df))

# ==========================================================
# CLASS WEIGHTS
# ==========================================================

weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(train_df["label"]),

    y=train_df["label"]

)

class_weights = {

    i: weights[i]

    for i in range(NUM_CLASSES)

}

print()

print(class_weights)

print("\nPart 1A Completed Successfully")



# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# ==========================================================
# DULL RAZOR HAIR REMOVAL
# ==========================================================

def remove_hair(image):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(17,17))

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, thresh = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

    result = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)

    return result


# ==========================================================
# CLAHE ENHANCEMENT
# ==========================================================

def apply_clahe(image):

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    l = clahe.apply(l)

    merged = cv2.merge((l,a,b))

    return cv2.cvtColor(merged,cv2.COLOR_LAB2RGB)


# ==========================================================
# IMAGE LOADER
# ==========================================================

def preprocess_image(path):

    path = path.decode("utf-8")

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype(np.float32) / 255.0

    return image

# ==========================================================
# TF WRAPPER
# ==========================================================

def tf_preprocess(path, label):

    image = tf.numpy_function(
        preprocess_image,
        [path],
        tf.float32
    )

    image.set_shape((IMG_SIZE, IMG_SIZE, 3))

    label = tf.cast(label, tf.int32)

    return image, label


# ==========================================================
# DATA AUGMENTATION
# ==========================================================

augment = tf.keras.Sequential([

    tf.keras.layers.RandomFlip("horizontal"),

    tf.keras.layers.RandomRotation(0.15),

    tf.keras.layers.RandomZoom(0.20),

    tf.keras.layers.RandomContrast(0.20),

    tf.keras.layers.RandomBrightness(0.20),

])


# ==========================================================
# DATASET CREATOR
# ==========================================================

def create_dataset(df,training=False):

    ds = tf.data.Dataset.from_tensor_slices(

        (

            df["image_path"].values,

            df["label"].values

        )

    )

    ds = ds.map(

        tf_preprocess,

        num_parallel_calls=AUTOTUNE

    )

    if training:

        ds = ds.shuffle(2000)

        ds = ds.map(

            lambda x,y:(augment(x),y),

            num_parallel_calls=AUTOTUNE

        )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(AUTOTUNE)

    return ds


# ==========================================================
# BUILD DATASETS
# ==========================================================

train_ds = create_dataset(train_df,True)

valid_ds = create_dataset(valid_df)

test_ds = create_dataset(test_df)

print("Datasets Created Successfully")


# ==========================================================
# VISUALIZE SAMPLE BATCH
# ==========================================================

plt.figure(figsize=(12,10))

for images,labels in train_ds.take(1):

    for i in range(9):

        plt.subplot(3,3,i+1)

        plt.imshow(images[i])

        plt.title(CLASS_NAMES[labels[i]])

        plt.axis("off")

plt.tight_layout()

plt.show()


# ==========================================================
# DATASET INFORMATION
# ==========================================================

print("="*60)

print("Training Images :",len(train_df))

print("Validation Images :",len(valid_df))

print("Testing Images :",len(test_df))

print("Image Size :",IMG_SIZE)

print("Batch Size :",BATCH_SIZE)

print("Classes :",NUM_CLASSES)

print("="*60)

print("\nPart 1B Completed Successfully")





# ==========================================================
# IMPORT MODEL LIBRARIES
# ==========================================================

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.applications import EfficientNetV2B3

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger
)

from tensorflow.keras.optimizers import Adam

# ==========================================================
# MODEL INPUT
# ==========================================================

INPUT_SHAPE = (224,224,3)

# ==========================================================
# LOAD PRETRAINED MODEL
# ==========================================================

base_model = EfficientNetV2B3(

    include_top=False,

    weights="imagenet",

    input_shape=INPUT_SHAPE

)

base_model.trainable = False

print("Base Model Loaded")

# ==========================================================
# BUILD CLASSIFIER
# ==========================================================

inputs = layers.Input(shape=INPUT_SHAPE)

x = base_model(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.40)(x)

x = layers.Dense(
    512,
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.30)(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(

    NUM_CLASSES,

    activation="softmax",

    dtype="float32"

)(x)

model = Model(inputs,outputs)

print(model.summary())



# ==========================================================
# COMPILE MODEL
# ==========================================================

optimizer = Adam(

    learning_rate=1e-4

)

model.compile(

    optimizer=optimizer,

    loss="sparse_categorical_crossentropy",

    metrics=[
    tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    tf.keras.metrics.SparseTopKCategoricalAccuracy(
        k=2,
        name="top2_accuracy"
    )
]

)

print("Model Compiled")



# ==========================================================
# CALLBACKS
# ==========================================================

checkpoint = ModelCheckpoint(

    MODEL_DIR/"best_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=3,

    verbose=1

)

csv_logger = CSVLogger(

    CSV_DIR/"training_log.csv"

)

callbacks = [

    checkpoint,

    early_stop,

    reduce_lr,

    csv_logger

]

print("Callbacks Ready")



# ==========================================================
# TRAIN MODEL
# ==========================================================

EPOCHS = 20

for images, labels in train_ds.take(1):
    print(labels)
    print(labels.shape)
    print(labels.dtype)
    
history = model.fit(

    train_ds,

    validation_data=valid_ds,

    epochs=EPOCHS,

    callbacks=callbacks,

    class_weight=class_weights,

    verbose=1

)


# ==========================================================
# FINE TUNING
# ==========================================================

print("="*60)
print("Starting Fine-Tuning...")
print("="*60)

base_model.trainable = True

# Freeze first layers
for layer in base_model.layers[:-100]:
    layer.trainable = False

optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-5
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=2,
            name="top2_accuracy"
        )
    ]
)

fine_history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)



# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(
    MODEL_DIR/"SkinNova_EfficientNetV2_Final.keras"
)

print("Model Saved Successfully")


# ==========================================================
# TEST ACCURACY
# ==========================================================

loss, accuracy, top2 = model.evaluate(
    test_ds,
    verbose=1
)

print()

print("Test Loss :",loss)

print("Test Accuracy :",accuracy)

print("Top2 Accuracy :",top2)



# ==========================================================
# PREDICTIONS
# ==========================================================

y_true = []
y_pred = []
y_prob = []

for images,labels in test_ds:

    pred = model.predict(images,verbose=0)

    y_true.extend(labels.numpy())

    y_pred.extend(np.argmax(pred,axis=1))

    y_prob.extend(pred)

y_true = np.array(y_true)

y_pred = np.array(y_pred)

y_prob = np.array(y_prob)

print("Prediction Completed")


from sklearn.metrics import classification_report

report = classification_report(

    y_true,

    y_pred,

    target_names=CLASS_NAMES,

    digits=4

)

print(report)

with open(
    OUTPUT_DIR/"classification_report.txt",
    "w"
) as f:

    f.write(report)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(9,8))

sns.heatmap(

    cm,

    annot=True,

    cmap="Blues",

    fmt="d",

    xticklabels=CLASS_NAMES,

    yticklabels=CLASS_NAMES

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(

    GRAPH_DIR/"confusion_matrix.png",

    dpi=300

)

plt.show()








plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"],label="Train")

plt.plot(history.history["val_accuracy"],label="Validation")

plt.legend()

plt.title("Accuracy")

plt.grid(True)

plt.savefig(

    GRAPH_DIR/"accuracy.png",

    dpi=300

)

plt.show()






plt.figure(figsize=(8,5))

plt.plot(history.history["loss"],label="Train")

plt.plot(history.history["val_loss"],label="Validation")

plt.legend()

plt.title("Loss")

plt.grid(True)

plt.savefig(

    GRAPH_DIR/"loss.png",

    dpi=300

)

plt.show()






history_df = pd.DataFrame(history.history)

history_df.to_csv(

    CSV_DIR/"training_history.csv",

    index=False

)

print("History Saved")








# ==========================================================
# RESEARCH METRICS
# ==========================================================

from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import label_binarize

print("Research Metrics Imported")






# ==========================================================
# BINARIZE LABELS
# ==========================================================

y_true_bin = label_binarize(
    y_true,
    classes=np.arange(NUM_CLASSES)
)

print(y_true_bin.shape)






# ==========================================================
# MULTI CLASS ROC
# ==========================================================

plt.figure(figsize=(10,8))

roc_auc = {}

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(
        y_true_bin[:,i],
        y_prob[:,i]
    )

    roc_auc[i] = auc(fpr,tpr)

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{CLASS_NAMES[i]} (AUC={roc_auc[i]:.3f})"
    )

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("Multi-Class ROC Curve")

plt.legend(fontsize=8)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR/"roc_curve.png",
    dpi=300
)

plt.show()



# ==========================================================
# PR CURVE
# ==========================================================

plt.figure(figsize=(10,8))

for i in range(NUM_CLASSES):

    precision, recall, _ = precision_recall_curve(
        y_true_bin[:,i],
        y_prob[:,i]
    )

    ap = average_precision_score(
        y_true_bin[:,i],
        y_prob[:,i]
    )

    plt.plot(
        recall,
        precision,
        label=f"{CLASS_NAMES[i]} AP={ap:.3f}"
    )

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.legend(fontsize=8)

plt.grid(True)

plt.savefig(
    GRAPH_DIR/"precision_recall_curve.png",
    dpi=300
)

plt.show()






# ==========================================================
# RESEARCH METRICS
# ==========================================================

accuracy = np.mean(y_true==y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

mcc = matthews_corrcoef(
    y_true,
    y_pred
)

kappa = cohen_kappa_score(
    y_true,
    y_pred
)

auc_score = roc_auc_score(
    y_true_bin,
    y_prob,
    multi_class="ovr"
)

print("="*60)

print("Accuracy :",accuracy)

print("Precision :",precision)

print("Recall :",recall)

print("F1 Score :",f1)

print("MCC :",mcc)

print("Kappa :",kappa)

print("ROC AUC :",auc_score)

print("="*60)









# ==========================================================
# SAVE METRICS
# ==========================================================

metrics = pd.DataFrame({

    "Accuracy":[accuracy],

    "Precision":[precision],

    "Recall":[recall],

    "F1 Score":[f1],

    "MCC":[mcc],

    "Cohen Kappa":[kappa],

    "ROC AUC":[auc_score]

})

metrics.to_csv(

    CSV_DIR/"research_metrics.csv",

    index=False

)

print(metrics)









# ==========================================================
# CLASS WISE ACCURACY
# ==========================================================

cm = confusion_matrix(y_true,y_pred)

class_accuracy = cm.diagonal()/cm.sum(axis=1)

class_df = pd.DataFrame({

    "Class":CLASS_NAMES,

    "Accuracy":class_accuracy

})

class_df.to_csv(

    CSV_DIR/"class_accuracy.csv",

    index=False

)

plt.figure(figsize=(8,5))

plt.bar(class_df["Class"],class_df["Accuracy"])

plt.xticks(rotation=45)

plt.ylabel("Accuracy")

plt.title("Class Wise Accuracy")

plt.tight_layout()

plt.savefig(

    GRAPH_DIR/"class_accuracy.png",

    dpi=300

)

plt.show()







# ==========================================================
# EXPLAINABLE AI
# ==========================================================

import matplotlib.cm as cm
import lime
import lime.lime_image

from skimage.segmentation import mark_boundaries

print("Explainable AI Libraries Loaded")






# ==========================================================
# FIND LAST CONV LAYER
# ==========================================================

for layer in reversed(base_model.layers):

    if isinstance(layer, tf.keras.layers.Conv2D):

        LAST_CONV_LAYER = layer.name

        break

print("Last Conv Layer :", LAST_CONV_LAYER)




# ==========================================================
# GRAD CAM
# ==========================================================

def make_gradcam_heatmap(img_array, model, last_conv_layer):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[...,tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap,0)

    heatmap /= tf.math.reduce_max(heatmap)

    return heatmap.numpy()





# ==========================================================
# DISPLAY GRADCAM
# ==========================================================

images, labels = next(iter(test_ds))

image = images[0]

img = tf.expand_dims(image,0)

heatmap = make_gradcam_heatmap(

    img,

    model,

    LAST_CONV_LAYER

)

heatmap = cv2.resize(

    heatmap,

    (224,224)

)

heatmap = np.uint8(255*heatmap)

heatmap = cm.jet(heatmap)[:,:,:3]

overlay = heatmap*0.4 + image.numpy()

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.imshow(image)

plt.title("Original")

plt.axis("off")

plt.subplot(1,2,2)

plt.imshow(overlay)

plt.title("Grad-CAM")

plt.axis("off")

plt.tight_layout()

plt.savefig(

    GRAPH_DIR/"gradcam_example.png",

    dpi=300

)

plt.show()









# ==========================================================
# LIME
# ==========================================================

explainer = lime.lime_image.LimeImageExplainer()

def predict_fn(images):

    return model.predict(images)

explanation = explainer.explain_instance(

    image.numpy(),

    predict_fn,

    top_labels=1,

    hide_color=0,

    num_samples=1000

)

temp, mask = explanation.get_image_and_mask(

    explanation.top_labels[0],

    positive_only=True,

    num_features=10,

    hide_rest=False

)

plt.figure(figsize=(6,6))

plt.imshow(

    mark_boundaries(temp,mask)

)

plt.title("LIME Explanation")

plt.axis("off")

plt.savefig(

    GRAPH_DIR/"lime.png",

    dpi=300

)

plt.show()










# ==========================================================
# SAMPLE PREDICTION
# ==========================================================

prediction = model.predict(img)

pred = np.argmax(prediction)

confidence = np.max(prediction)

print()

print("Prediction :",CLASS_NAMES[pred])

print("Confidence :",round(confidence*100,2),"%")

print("Actual :",CLASS_NAMES[labels[0]])





# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

prediction_df = pd.DataFrame({

    "Actual":y_true,

    "Prediction":y_pred

})

prediction_df["Actual"] = encoder.inverse_transform(

    prediction_df["Actual"]

)

prediction_df["Prediction"] = encoder.inverse_transform(

    prediction_df["Prediction"]

)

prediction_df.to_csv(

    CSV_DIR/"predictions.csv",

    index=False

)

print("Prediction CSV Saved")








# ==========================================================
# CORRECT VS INCORRECT
# ==========================================================

plt.figure(figsize=(15,10))

count = 1

for images,labels in test_ds.take(3):

    preds = np.argmax(model.predict(images,verbose=0),axis=1)

    for i in range(len(images)):

        if count > 12:
            break

        plt.subplot(3,4,count)

        plt.imshow(images[i])

        color = "green" if preds[i]==labels[i] else "red"

        plt.title(

            f"P:{CLASS_NAMES[preds[i]]}\nT:{CLASS_NAMES[labels[i]]}",

            color=color,

            fontsize=8

        )

        plt.axis("off")

        count += 1

plt.tight_layout()

plt.savefig(

    GRAPH_DIR/"prediction_gallery.png",

    dpi=300

)

plt.show()






# ==========================================================
# FINISH
# ==========================================================

print("="*70)
print(" SkinNova Training Completed Successfully ")
print("="*70)

print("\nSaved Outputs:")

print("✔ Trained Model")
print("✔ Confusion Matrix")
print("✔ ROC Curve")
print("✔ Precision Recall Curve")
print("✔ Accuracy Graph")
print("✔ Loss Graph")
print("✔ Class Accuracy")
print("✔ Grad-CAM")
print("✔ LIME")
print("✔ Prediction Gallery")
print("✔ CSV Reports")
print("✔ Classification Report")
print("✔ Research Metrics")