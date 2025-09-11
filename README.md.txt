# Animal Classifier App

This is a multi-step Streamlit app to classify animals using images. 
It supports multiple datasets, real-time analysis, OOD detection for unknown classes, 
and provides downloadable CSV and PDF reports.

## Features
- Step 1: Add multiple datasets of animals
- Step 2: Preview and manage datasets
- Step 3: Train CNN model with accuracy vs epochs chart
- Step 4: Predict new images with grid view & downloadable results
- Unknown class detection using Autoencoder
- Professional UI with stylized steps

## Requirements
See `requirements.txt` for dependencies.

## How to run
```bash
streamlit run classifier_app.py
