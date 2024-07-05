# Face Recognition App

## Overview

This is a simple face recognition application using OpenCV and Tkinter. It provides two main functionalities: **Sign In** and **Sign Up**. Users can capture their face data for recognition or use the captured data to recognize faces from the webcam.

## Features

- **Sign In**: Captures face frames from the webcam and saves them to a `.npy` file.
- **Sign Up**: Recognizes faces from the webcam based on saved data.

## Installation

You can run this application either using Docker or directly on your local machine. 

### With Docker

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/your-repository.git
    cd your-repository
    ```

2. **Build the Docker image:**

    ```sh
    docker build -t face-recognition-app .
    ```

3. **Run the Docker container:**

    ```sh
    docker run -it --rm face-recognition-app
    ```

   This will start the Tkinter GUI application inside the Docker container.

### Without Docker

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/your-repository.git
    cd your-repository
    ```

2. **Create a virtual environment (optional but recommended):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Run the application:**

    ```sh
    python app.py
    ```

   This will start the Tkinter GUI application on your local machine.

## How to Use the App

1. **Sign In**:
   - Click the **"Sign In"** button.
   - Enter your name when prompted.
   - Move your head around and adjust your distance from the camera to capture frames.
   - The captured frames will be saved in the `faces` directory as a `.npy` file with your name.

2. **Sign Up**:
   - Click the **"Sign Up"** button.
   - The app will start recognizing faces using the captured face data.
   - The app will display the name of the recognized person or "undefined" if the face is not recognized.

## Dependencies

The dependencies for this project are listed in `requirements.txt`. You can install them using the following command:

```sh
pip install -r requirements.txt
