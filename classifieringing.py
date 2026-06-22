import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Animal Classifier", layout="wide")

st.markdown(
    """
    <style>
    .main-title {font-size:36px; font-weight:bold; color:#2E86C1; text-align:center; margin-bottom:20px;}
    .step-header {font-size:24px; font-weight:bold; color:#1A5276; margin-top:20px;}
    .card {padding:20px; border-radius:12px; background-color:#f8f9fa; box-shadow:0 2px 8px rgba(0,0,0,0.1); margin-bottom:20px;}
    .stButton>button {border-radius:8px; background:#2E86C1; color:white; font-weight:bold;}
    .delete-button>button {background:#C0392B !important; color:white !important; font-weight:bold;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>🐾 Animal Classifier App</div>", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "datasets" not in st.session_state:
    st.session_state.datasets = {}
if "model" not in st.session_state:
    st.session_state.model = None
if "class_name" not in st.session_state:
    st.session_state.class_name = ""
if "files" not in st.session_state:
    st.session_state.files = []
if "predictions" not in st.session_state:
    st.session_state.predictions = pd.DataFrame()
if "history" not in st.session_state:
    st.session_state.history = None
if "class_labels" not in st.session_state:
    st.session_state.class_labels = []

# -------------------- RESET --------------------
def reset_all():
    st.session_state.datasets = {}
    st.session_state.model = None
    st.session_state.class_name = ""
    st.session_state.files = []
    st.session_state.predictions = pd.DataFrame()
    st.session_state.history = None
    st.session_state.class_labels = []

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📥 Step 1: Upload Datasets", "📂 Step 2: Review Datasets",
                                  "⚙️ Step 3: Train Model", "🔮 Step 4: Predict"])

# -------------------- STEP 1 --------------------
with tab1:
    st.markdown("<div class='step-header'>📥 Step 1: Upload Datasets</div>", unsafe_allow_html=True)

    st.session_state.class_name = st.text_input("Enter Class Name (e.g., Dog, Cat)", value=st.session_state.class_name)
    st.session_state.files = st.file_uploader(
        "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="file_uploader_step1"
    )

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("➕ Add Dataset"):
            if st.session_state.class_name and st.session_state.files:
                images = []
                for file in st.session_state.files:
                    img = Image.open(file).convert("RGB").resize((128, 128))
                    images.append(np.array(img) / 255.0)
                st.session_state.datasets[st.session_state.class_name] = (
                    st.session_state.datasets.get(st.session_state.class_name, []) + images
                )
                st.success(f"✅ Added {len(st.session_state.files)} images to class: {st.session_state.class_name}")
                st.balloons()
            else:
                st.error("⚠️ Please enter a class name and upload images.")
    with col2:
        if st.button("🧹 Clear Uploads"):
            st.session_state.class_name = ""
            st.session_state.files = []
            st.rerun()

    if st.session_state.datasets:
        st.info("✅ Datasets added successfully. Proceed to Step 2.")

# -------------------- STEP 2 --------------------
with tab2:
    st.markdown("<div class='step-header'>📂 Step 2: Review Datasets</div>", unsafe_allow_html=True)

    if st.session_state.datasets:
        for cname, imgs in list(st.session_state.datasets.items()):
            with st.expander(f"📁 {cname} ({len(imgs)} images)"):
                cols = st.columns(5)
                for i, img in enumerate(imgs):
                    with cols[i % 5]:
                        st.image(img, use_container_width=True)

                if st.button(f"❌ Delete {cname}", key=f"delete_{cname}"):
                    del st.session_state.datasets[cname]
                    st.rerun()
    else:
        st.warning("⚠️ No datasets found. Please add in Step 1.")

# -------------------- STEP 3 --------------------
with tab3:
    st.markdown("<div class='step-header'>⚙️ Step 3: Train Model (Real-Time Analysis + PDF Report)</div>", unsafe_allow_html=True)

    if len(st.session_state.datasets) >= 2:
        if st.button("🚀 Start Training"):
            X, y, class_labels = [], [], list(st.session_state.datasets.keys())
            st.session_state.class_labels = class_labels
            for idx, cname in enumerate(class_labels):
                for img in st.session_state.datasets[cname]:
                    X.append(img)
                    y.append(idx)
            X, y = np.array(X), tf.keras.utils.to_categorical(y, num_classes=len(class_labels))

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            model = models.Sequential([
                layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
                layers.MaxPooling2D(2,2),
                layers.Conv2D(64, (3,3), activation="relu"),
                layers.MaxPooling2D(2,2),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(len(class_labels), activation="softmax")
            ])
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            # Real-time placeholders
            progress_bar = st.progress(0)
            status_text = st.empty()
            acc_chart = st.line_chart({"Train Acc": [], "Val Acc": []}, height=200)
            loss_chart = st.line_chart({"Train Loss": [], "Val Loss": []}, height=200)

            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = int((epoch + 1) / self.params['epochs'] * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"📊 Epoch {epoch+1}/{self.params['epochs']} "
                                     f"- Acc: {logs['accuracy']:.2f}, Val_Acc: {logs['val_accuracy']:.2f}, "
                                     f"Loss: {logs['loss']:.2f}, Val_Loss: {logs['val_loss']:.2f}")
                    acc_chart.add_rows({"Train Acc": [logs["accuracy"]], "Val Acc": [logs["val_accuracy"]]})
                    loss_chart.add_rows({"Train Loss": [logs["loss"]], "Val Loss": [logs["val_loss"]]})

            history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val),
                                verbose=0, callbacks=[StreamlitCallback()])

            st.session_state.model = model
            st.session_state.history = history
            st.success("✅ Training completed!")

            # -------- Final Static Plots --------
            st.markdown("### 📊 Final Training Curves")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

            ax1.plot(history.history["accuracy"], label="Train Acc")
            ax1.plot(history.history["val_accuracy"], label="Val Acc")
            ax1.set_title("Accuracy vs Epochs")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Accuracy")
            ax1.legend()

            ax2.plot(history.history["loss"], label="Train Loss")
            ax2.plot(history.history["val_loss"], label="Val Loss")
            ax2.set_title("Loss vs Epochs")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Loss")
            ax2.legend()

            st.pyplot(fig)

            # Save plots for PDF
            acc_loss_img = io.BytesIO()
            fig.savefig(acc_loss_img, format="png")
            acc_loss_img.seek(0)

            # Confusion Matrix
            y_pred = np.argmax(model.predict(X_val), axis=1)
            y_true = np.argmax(y_val, axis=1)
            cm = confusion_matrix(y_true, y_pred)

            st.markdown("### 📌 Confusion Matrix")
            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(np.arange(len(class_labels)))
            ax.set_yticks(np.arange(len(class_labels)))
            ax.set_xticklabels(class_labels)
            ax.set_yticklabels(class_labels)
            plt.colorbar(im, ax=ax)
            for i in range(len(class_labels)):
                for j in range(len(class_labels)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
            st.pyplot(fig)

            cm_img = io.BytesIO()
            fig.savefig(cm_img, format="png")
            cm_img.seek(0)

            # -------- PDF Report --------
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("🐾 Animal Classifier - Training Report", styles['Title']))
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("📊 Training Summary", styles['Heading2']))
            elements.append(Paragraph(f"Final Training Accuracy: {history.history['accuracy'][-1]:.2f}", styles['Normal']))
            elements.append(Paragraph(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2f}", styles['Normal']))
            elements.append(Paragraph(f"Final Training Loss: {history.history['loss'][-1]:.2f}", styles['Normal']))
            elements.append(Paragraph(f"Final Validation Loss: {history.history['val_loss'][-1]:.2f}", styles['Normal']))
            elements.append(Spacer(1, 20))

            elements.append(Paragraph("📈 Accuracy & Loss Curves", styles['Heading2']))
            elements.append(RLImage(acc_loss_img, width=400, height=200))
            elements.append(Spacer(1, 20))

            elements.append(Paragraph("📌 Confusion Matrix", styles['Heading2']))
            elements.append(RLImage(cm_img, width=400, height=200))

            doc.build(elements)
            pdf_buffer.seek(0)

            st.download_button("📥 Download PDF Report", data=pdf_buffer,
                               file_name="training_report.pdf", mime="application/pdf")

    else:
        st.warning("⚠️ Please add at least 2 datasets before training.")

# -------------------- STEP 4 --------------------
with tab4:
    st.markdown("<div class='step-header'>🔮 Step 4: Predict & Final Analysis</div>", unsafe_allow_html=True)

    if st.session_state.model:
        test_files = st.file_uploader("Upload Images for Prediction", type=["jpg","jpeg","png"], accept_multiple_files=True)
        if test_files:
            preds_data = []
            cols = st.columns(5)
            for i, test_file in enumerate(test_files):
                img = Image.open(test_file).convert("RGB").resize((128,128))
                img_array = np.expand_dims(np.array(img)/255.0, axis=0)
                preds = st.session_state.model.predict(img_array, verbose=0)
                class_idx = np.argmax(preds)
                confidence = np.max(preds)

                with cols[i % 5]:
                    st.image(img, caption=f"{st.session_state.class_labels[class_idx]} ({confidence*100:.1f}%)", use_container_width=True)

                preds_data.append({
                    "Filename": test_file.name,
                    "Prediction": st.session_state.class_labels[class_idx],
                    "Confidence": round(confidence*100, 2)
                })

            st.session_state.predictions = pd.DataFrame(preds_data)

            st.markdown("### 📑 Final Analysis")
            st.dataframe(st.session_state.predictions)

            csv_buffer = io.StringIO()
            st.session_state.predictions.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Results as CSV", csv_buffer.getvalue(), "predictions.csv", "text/csv")

    else:
        st.warning("⚠️ Please train the model in Step 3 first.")

# -------------------- RESET BUTTON --------------------
st.sidebar.markdown("### 🔄 Controls")
if st.sidebar.button("Reset App"):
    reset_all()
    st.rerun()
