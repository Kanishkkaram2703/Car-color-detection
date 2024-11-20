import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained car color detection model
model = load_model(r"F:/KANISHK/projects_null_class/car_color/car_color_classification_model.keras")  # Use raw string or forward slashes
class_labels = [
    "beige", "black", "blue", "brown", "gold", "green", "grey", "orange", 
    "pink", "purple", "red", "silver", "tan", "white", "yellow"
]

# Load Haar cascades
person_cascade = cv2.CascadeClassifier(r"F:/KANISHK/projects_null_class/car_color/haarcascade_fullbody.xml")
car_cascade = cv2.CascadeClassifier(r"F:/KANISHK/projects_null_class/car_color/haarcascade_car.xml")

# Image preprocessing for prediction
def prepare_image(image):
    img_height, img_width = 224, 224  # Match the model input size
    image = cv2.resize(image, (img_width, img_height))
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to process video or webcam feed
def process_feed(video_source):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        people = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in people:
            # Draw green rectangles for people
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Detect cars
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in cars:
            # Extract car region of interest (ROI)
            car_roi = frame[y:y + h, x:x + w]
            processed_image = prepare_image(car_roi)
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction)
            car_color = class_labels[predicted_class]

            # Choose rectangle color based on car color
            if car_color == "blue":
                rectangle_color = (0, 0, 255)  # Red for blue cars
            else:
                rectangle_color = (255, 0, 0)  # Blue for other cars

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
            cv2.putText(frame, car_color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rectangle_color, 2)

        # Display the frame
        cv2.imshow("Car Color and Person Detection", frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Implementation with Tkinter
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        process_feed(file_path)

def start_webcam():
    process_feed(0)

# Create the GUI window
root = tk.Tk()
root.title("Car Color and Person Detection")
root.geometry("400x200")

tk.Label(root, text="Car Color and Person Detection", font=("Arial", 16)).pack(pady=10)
tk.Button(root, text="Select Video File", command=open_file, width=20, height=2).pack(pady=10)
tk.Button(root, text="Start Webcam", command=start_webcam, width=20, height=2).pack(pady=10)
tk.Button(root, text="Exit", command=root.quit, width=20, height=2).pack(pady=10)

root.mainloop()
    