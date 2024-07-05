import cv2
import numpy as np
import os

# Initialize the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load face data
def load_face_data():
    face_data = {}
    for filename in os.listdir('faces'):
        if filename.endswith('.npy'):
            name = filename.split('.')[0]
            face_data[name] = np.load(os.path.join('faces', filename))
    return face_data

# Recognize face
def recognize_face(face, face_data, face_size=(100, 100), threshold=2000):
    face_resized = cv2.resize(face, face_size)
    min_dist = float('inf')
    recognized_name = 'undefined'
    
    for name, data in face_data.items():
        # Compute the mean face vector for each person
        mean_face = np.mean(data, axis=0)
        dist = np.linalg.norm(face_resized - mean_face)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            recognized_name = name

    return recognized_name

# Main function
def main():
    face_data = load_face_data()
    
    # Initialize video capture from the default camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw a rectangle around each face and recognize it
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_region = gray[y:y+h, x:x+w]
            recognized_name = recognize_face(face_region, face_data)
            cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
