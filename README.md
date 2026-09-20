# HandWriteAI

HandWriteAI is a handwritten character recognition system built using a Convolutional Neural Network (CNN).

The project uses the EMNIST Balanced dataset and provides a FastAPI backend for predicting handwritten characters from uploaded images.

---

## Features

- Handwritten character recognition
- EMNIST Balanced dataset
- 47 character classes
- CNN-based image classification
- Image preprocessing using OpenCV
- FastAPI REST API
- Confidence score
- Top-5 predictions
- JSON class-to-character mapping

---

## Project Architecture

```text
HandWriteAI
│
├── dataset/
│   └── emnist/
│       ├── emnist-balanced-train.csv
│       └── emnist-balanced-test.csv
│
├── model/
│   └── emnist_cnn.keras
│
├── notebook/
│   └── emnist_cnn.ipynb
│
├── app/
│   └── Backend/
│       ├── main.py
│       ├── emnist_cnn.keras
│       ├── result.json
│       └── debug_processed.png
│
├── requirements.txt
└── README.md