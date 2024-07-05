import cv2
import numpy as np
import os

# Ensure the 'faces' directory exists
if not os.path.exists('faces'):
    os.makedirs('faces')

# Initialize the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def capture_face_frames(name, total_frames=300, face_size=(100, 100)):
    # Initialize video capture from the default camera
    cap = cv2.VideoCapture(0)
    frames = []
    frame_count = 0

    print("Starting face capture. Please move your head around and adjust your distance from the camera.")

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        # Draw a rectangle around each face and resize the face region
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_region = gray[y:y+h, x:x+w]
            face_region_resized = cv2.resize(face_region, face_size)
            frames.append(face_region_resized)
            frame_count += 1

        # Show the frame counter on the video feed
        cv2.putText(frame, f"Frames Taken: {frame_count}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame with the rectangle(s)
        cv2.imshow('Face Capture', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

    # Save the frames to a .npy file
    frames = np.array(frames)
    np.save(os.path.join('faces', f'{name}.npy'), frames)
    print(f'Saved {len(frames)} frames to {name}.npy')

if __name__ == "__main__":
    user_name = input("Enter your name: ")
    capture_face_frames(user_name)
