# Dog Breed Recognition — Hybrid CNN Model (ConvNeXtSmall + EfficientNetV2B0)

A deep learning model that classifies dog images into **120 breeds** using the
[Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset).
It uses a **hybrid transfer-learning architecture**, combining features from two
pretrained backbones — **ConvNeXtSmall** and **EfficientNetV2B0** — to improve
classification accuracy over either model alone.

This model was built as the breed-recognition component of **Canine Mart**, a
final-year BS IT project combining AI dog-breed recognition, a Generative AI
advisory chatbot, and a moderated e-commerce marketplace.

## Overview

- **Task:** Multi-class image classification (120 dog breeds)
- **Dataset:** Stanford Dogs Dataset (Kaggle), ~20,000 images with XML annotations
- **Architecture:** Hybrid dual-input model — features from ConvNeXtSmall and
  EfficientNetV2B0 (both pretrained on ImageNet) are combined for the final
  classification head
- **Framework:** TensorFlow / Keras
- **Result:** ~91% validation accuracy
- **Training environment:** Google Colab (free GPU tier)

## Pipeline

1. **Data loading** — downloads the dataset via `kagglehub` and parses breed
   labels from the Pascal-VOC style XML annotations.
2. **Preprocessing** — encodes labels with `LabelEncoder`, splits data into
   train / validation / test sets (80/10/10, stratified).
3. **Data visualization** — class distribution plot and sample image previews.
4. **Augmentation** — random flip, rotation, zoom, and contrast via
   `keras.Sequential` preprocessing layers.
5. **tf.data pipeline** — images loaded and resized (224×224) using
   `tf.py_function` + OpenCV, batched for training.
6. **Model** — two frozen pretrained backbones (ConvNeXtSmall,
   EfficientNetV2B0) feed a shared classification head trained on top.
7. **Training** — Adam optimizer, sparse categorical cross-entropy, 15 epochs.
8. **Evaluation** — accuracy/loss curves plotted; final test set evaluation.
9. **Model export** — saved as a `.keras` file (originally to Google Drive).

## Tech Stack

- Python, TensorFlow, Keras
- OpenCV, NumPy, Pandas, Matplotlib
- scikit-learn (`LabelEncoder`, `train_test_split`)
- kagglehub (dataset download)

## Repository Contents

- `Hybrid_model_complete.ipynb` — full notebook: data loading, preprocessing,
  training, and evaluation.

## How to Run

1. Open the notebook in Google Colab or Jupyter.
2. Install dependencies:
   ```bash
   pip install tensorflow keras opencv-python lxml kagglehub
   ```
3. Run the cells in order. The notebook downloads the Stanford Dogs Dataset
   automatically via `kagglehub`.
4. Training requires a GPU runtime (Colab's free GPU tier works fine).
5. The trained model is saved as a `.keras` file at the end of the notebook —
   update the save path if not using Google Drive/Colab.

## Project Context

This model powers the breed-recognition feature of **Canine Mart**, where it's
integrated into a Flask REST API backend and used alongside a Generative
AI–powered advisory chatbot to help users identify and learn about dog breeds
within a moderated marketplace app.

## License

This project is for educational/portfolio purposes. The Stanford Dogs Dataset
is provided under its own license — see the
[dataset page](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
for details.
