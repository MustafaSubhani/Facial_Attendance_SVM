# 📸 Facial Attendance System

This project is a real-time Facial Recognition-based Attendance System that leverages a pre-trained FaceNet deep learning model for generating high-quality face embeddings, and a Support Vector Machine (SVM) classifier for identifying people. The object detection portion is powered seamlessly by MTCNN. It provides an automated attendance logging workflow and stores the logs seamlessly inside a CSV file.

## ✨ Features

- **Real-Time Detection & Recognition**: Continuously process video streams via webcam to detect and recognize faces on the fly.
- **State-of-the-Art Deep Learning Models**: 
  - **MTCNN**: Used for robust face detection and perfect bounding box coordinate placement.
  - **FaceNet (InceptionResnetV1)**: Extracts consistent 512-dimensional facial feature embeddings, utilizing weights heavily pre-trained on the `vggface2` dataset.
- **Fast Classification**: Uses a lightweight SVM machine learning model trained directly on the derived facial embeddings for uncompromised inference latency.
- **Attendance Logging**: Automatically records the attendance with explicit timestamps and stores the results cleanly inside `attendance.csv`.

## 📊 Model Performance & Metrics

Based on the training artifacts generated from our custom training notebooks using the dataset:
- **Faces Evaluated**: Validated and generated embedding weights for a total of **118 faces**.
- **Model Accuracy**: Our core SVM classifier successfully achieved an accuracy rate of **93.22%** when training to evaluate these specific embeddings map instances.

## 📁 Project Structure

```
Facial_Attendance_SVM/
│
├── src/
│   └── webcam_attendance.py         # Main runtime Python script for webcam inference
├── models/
│   ├── svm_model.pkl                # Serialized pre-trained SVM model
│   └── label_encoder.pkl            # Associated deterministic label encoder
├── notebooks/
│   └── Class_Attendance_System_Model_Training.ipynb   # Jupyter Notebook used to conduct pipeline building and training
├── data/                            # Contains private datasets or data links (ignored in version control)
├── .gitignore
└── README.md
```

## 🔗 Dataset Access

> **Note:** The dataset used to orchestrate the training architecture inside this facial recognition deployment suite is **custom and strictly private**. As such, it is omitted from this repository and cannot be shared publicly.

## 🛠️ Setup & Local Installation

### Prerequisites

Ensure you have Python 3.7+ available inside your working environment. Follow this up by verifying you have the following packages properly sourced and installed:
```bash
pip install opencv-python numpy torch torchvision facenet-pytorch scikit-learn pandas joblib
```

### Running the Project Workflow

1. Navigate your terminal to the root repository folder.
2. **Execute the Webcam Instance:** Start the inference program by launching the execution script via Python.
    ```bash
    python src/webcam_attendance.py
    ```
3. **Real-time Evaluation:** Step in front of the camera! Watch as the green bounding box registers tracking mechanisms over your face correctly labeled by string format! Keep in mind you can easily stop the running application loop by hitting `q` on your active keyboard.
4. **Attendance Validation:** Open up the generated `attendance.csv` file created in the root directory matching faces recognized alongside correct session execution date-times securely labeled as `'Present'`.
