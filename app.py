import tkinter as tk
from tkinter import simpledialog
import cv2
import numpy as np
import os


class FaceCaptureApp:
    """
    A GUI application for face recognition with 'Sign In' and 'Sign Up' functionalities.
    """
    def __init__(self, master):
        """
        Initializes the FaceCaptureApp with the main Tkinter window and buttons.
        """
        self.master = master
        self.master.title("Face Recognition App")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Ensure the 'faces' directory exists
        if not os.path.exists('faces'):
            os.makedirs('faces')

        # Create and place the 'Sign In' button
        self.sign_in_button = tk.Button(self.master, text="Sign In", command=self.sign_in)
        self.sign_in_button.pack(pady=10)

        # Create and place the 'Sign Up' button
        self.sign_up_button = tk.Button(self.master, text="Sign Up", command=self.sign_up)
        self.sign_up_button.pack(pady=10)

    def capture_face_frames(self, name, total_frames=300, face_size=(100, 100)):
        """
        Captures face frames from the webcam and saves them to a .npy file.
        """
        cap = cv2.VideoCapture(0)
        frames = []
        frame_count = 0

        print("Starting face capture. Please move your head around and adjust your distance from the camera.")

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_region = gray[y:y+h, x:x+w]
                face_region_resized = cv2.resize(face_region, face_size)
                frames.append(face_region_resized)
                frame_count += 1

            cv2.putText(frame, f"Frames Taken: {frame_count}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Face Capture', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        frames = np.array(frames)
        np.save(os.path.join('faces', f'{name}.npy'), frames)
        print(f'Saved {len(frames)} frames to {name}.npy')

    def recognize_face(self, face, face_data, face_size=(100, 100), threshold=2000):
        """
        Recognizes a face from the loaded face data based on the minimum distance metric.
        """
        face_resized = cv2.resize(face, face_size)
        min_dist = float('inf')
        recognized_name = 'undefined'
        
        for name, data in face_data.items():
            mean_face = np.mean(data, axis=0)
            dist = np.linalg.norm(face_resized - mean_face)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                recognized_name = name

        return recognized_name

    def load_face_data(self):
        """
        Loads face data from the 'faces' directory.
        """
        face_data = {}
        for filename in os.listdir('faces'):
            if filename.endswith('.npy'):
                name = filename.split('.')[0]
                face_data[name] = np.load(os.path.join('faces', filename))
        return face_data

    def main_recognition(self):
        """
        Main function for face recognition.
        """
        face_data = self.load_face_data()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_region = gray[y:y+h, x:x+w]
                recognized_name = self.recognize_face(face_region, face_data)
                cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def sign_in(self):
        """
        Handles the 'Sign In' button action.
        """
        user_name = simpledialog.askstring("Input", "Enter your name:")
        if user_name:
            self.capture_face_frames(user_name)

    def sign_up(self):
        """
        Handles the 'Sign Up' button action.
        """
        self.main_recognition()


# Create the main Tkinter window
root = tk.Tk()
app = FaceCaptureApp(root)
root.mainloop()
