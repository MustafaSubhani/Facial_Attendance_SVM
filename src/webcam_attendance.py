import cv2
import numpy as np
import torch
import joblib
from datetime import datetime
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

# Load the trained SVM model and label encoder
svm_classifier = joblib.load('models/svm_model.pkl')  # Load your saved SVM classifier model
le = joblib.load('models/label_encoder.pkl')  # Load your saved Label Encoder

# Initialize MTCNN face detector and FaceNet model
mtcnn = MTCNN()
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define image transformation pipeline (resize, to tensor, normalize)
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert image from NumPy to PIL Image
    transforms.Resize((160, 160)),  # Resize the image to 160x160
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to range [-1, 1]
])

# Initialize the webcam
cap = cv2.VideoCapture(0)

# To store attendance
attendance_set = set()
attendance_data = []

# Define the function to extract embeddings and predict labels for all faces
def predict_labels(img):
    faces, _ = mtcnn.detect(img)
    results = []
    if faces is not None:
        for face in faces:
            # Crop face and generate embedding
            cropped_face = img[int(face[1]):int(face[3]), int(face[0]):int(face[2])]
            
            # Convert cropped face to PyTorch tensor
            cropped_face_tensor = transform(cropped_face).unsqueeze(0)  # Apply transformation and add batch dimension
            
            # Ensure the model is in evaluation mode
            model.eval()
            
            # Generate face embedding
            with torch.no_grad():  # No need to compute gradients during inference
                face_embedding = model(cropped_face_tensor).detach().cpu().numpy().flatten()  # Generate face embedding
            
            # Predict label
            prediction = svm_classifier.predict([face_embedding])
            predicted_label = le.inverse_transform(prediction)[0]
            results.append((predicted_label, face))
    return results

# Start webcam capture and recognition
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally (optional)
    frame = cv2.flip(frame, 1)
    
    # Predict labels for all detected faces
    predictions = predict_labels(frame)
    
    for label, face_coords in predictions:
        # Draw rectangle around the detected face
        x1, y1, x2, y2 = map(int, face_coords)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display the label on top of the rectangle
        display_label = label if label else "Unknown"
        cv2.putText(frame, display_label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        
        # If a label is predicted and not already in the attendance set, record attendance
        if label and label not in attendance_set:
            attendance_set.add(label)  # Add to the attendance set
            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            attendance_data.append([label, time_now, 'Present'])
            print(f"Recognized: {label}, Time: {time_now}")
    
    # Display the frame with the labels
    cv2.imshow("Webcam - Face Recognition", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the webcam and the OpenCV window
cap.release()
cv2.destroyAllWindows()

# Create a DataFrame and save to CSV
df = pd.DataFrame(attendance_data, columns=["Label", "Time", "Status"])
df.to_csv("attendance.csv", index=False)
print("Attendance saved to 'attendance.csv'.")