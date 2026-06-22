import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ------------------- App Setup -------------------
st.title("Dynamic Wildlife Image Classifier")
st.write("""
Upload images class by class, train the model, 
and then predict new unseen images with confidence scores.
""")

# ------------------- Folders -------------------
TRAIN_FOLDER = "train_images"
MODEL_FOLDER = "model"
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ------------------- Session State -------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "text_key" not in st.session_state:
    st.session_state.text_key = 0

# ------------------- Step 1: Upload Images for a Class -------------------
st.subheader("Step 1: Upload Images for a Class")

uploaded_files = st.file_uploader(
    "Upload images (jpg, jpeg, png) for this class",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}"
)

class_name = st.text_input(
    "Enter class/animal name for these images",
    key=f"class_input_{st.session_state.text_key}"
)

if st.button("Add Images to Dataset"):
    if uploaded_files and class_name.strip() != "":
        class_folder = os.path.join(TRAIN_FOLDER, class_name.strip())
        os.makedirs(class_folder, exist_ok=True)
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            img.save(os.path.join(class_folder, file.name))
        st.success(f"Added {len(uploaded_files)} images to class '{class_name.strip()}'")

        # ---------------- RESET uploader and text input ----------------
        st.session_state.uploader_key += 1
        st.session_state.text_key += 1
    else:
        st.error("Please upload images and enter a valid class name.")

# ------------------- Step 2: Train Model -------------------
st.subheader("Step 2: Train Model")

BATCH_SIZE = 16
EPOCHS = 10
IMG_SIZE = (64, 64)
SECONDS_PER_BATCH_ESTIMATE = 0.5  # Rough estimate

if st.button("Estimate Training Time"):
    total_images = sum([len(files) for r, d, files in os.walk(TRAIN_FOLDER)])
    if total_images == 0:
        st.warning("No images found. Upload some images first.")
    else:
        steps_per_epoch = total_images / BATCH_SIZE
        estimated_time = steps_per_epoch * EPOCHS * SECONDS_PER_BATCH_ESTIMATE
        st.info(f"Estimated training time: ~{estimated_time:.1f} seconds")

if st.button("Start Training"):
    if not any(os.scandir(TRAIN_FOLDER)):
        st.warning("No images found to train. Upload images first.")
    else:
        st.write("Training started...")

        # Data generator
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

        train_generator = datagen.flow_from_directory(
            TRAIN_FOLDER,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            color_mode='rgb'
        )

        val_generator = datagen.flow_from_directory(
            TRAIN_FOLDER,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            color_mode='rgb'
        )

        # Build CNN
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1],3)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(len(train_generator.class_indices), activation='softmax')
        ])

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        with st.spinner('Training in progress...'):
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=EPOCHS
            )

        # Save model
        model.save(os.path.join(MODEL_FOLDER, "trained_model.h5"))
        st.success("Training completed! Model saved as trained_model.h5")

        # ------------------- Confusion Matrix -------------------
        st.subheader("Confusion Matrix")
        val_generator.reset()
        Y_pred = model.predict(val_generator)
        y_pred = np.argmax(Y_pred, axis=1)
        y_true = val_generator.classes
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(train_generator.class_indices.keys()))

        fig, ax = plt.subplots(figsize=(8,6))
        disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
        st.pyplot(fig)

# ------------------- Step 3: Predict New Images -------------------
st.subheader("Step 3: Predict New Images")
uploaded_test_images = st.file_uploader(
    "Upload one or more test images (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_test_images and os.path.exists(os.path.join(MODEL_FOLDER, "trained_model.h5")):
    model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, "trained_model.h5"))
    class_labels = list(os.listdir(TRAIN_FOLDER))

    st.write("Predictions:")

    for uploaded_image in uploaded_test_images:
        image = Image.open(uploaded_image).convert("RGB").resize(IMG_SIZE)
        image_array = np.expand_dims(np.array(image)/255.0, axis=0)
        prediction = model.predict(image_array)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index] * 100

        st.image(image, caption=f"Uploaded Image: {uploaded_image.name}", use_column_width=True)
        st.write(f"Predicted Class: **{class_labels[class_index]}** with confidence: **{confidence:.2f}%**")
