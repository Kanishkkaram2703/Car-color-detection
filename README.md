# Car-color-detection

This project implements a machine learning model to detect vehicles, identify their colors, and count nearby people from images or video feeds. The system provides real-time annotations by drawing colored rectangles based on the detected object.

Features
Vehicle Color Detection: Recognizes the color of vehicles from a predefined set of colors (e.g., yellow, red, blue, etc.).
Dynamic Annotations:
Blue Rectangle: Drawn around vehicles of colors other than "blue."
Red Rectangle: Drawn around vehicles detected as "blue."
Green Rectangle: Drawn around detected people, including their count.
Haar Cascade Integration: Detects vehicles and people using pre-trained Haar cascades.
GUI Interface: Built using Tkinter to allow users to process video files or live webcam feeds.
Dataset
The project uses the VCOR - Vehicle Color Recognition Dataset from Kaggle, which provides labeled images of vehicles in various colors.

Requirements
Python 3.8 or higher
OpenCV
TensorFlow/Keras
NumPy
Tkinter (default with Python)
Pre-trained Haar cascades (haarcascade_car.xml, haarcascade_fullbody.xml)
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/vehicle-color-recognition.git
cd vehicle-color-recognition
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Download and extract the dataset from Kaggle:

VCOR - Vehicle Color Recognition Dataset
Place the dataset in a folder named dataset within the project directory.

Place the .keras model file:

Due to its large size, the trained .keras model is not included in this repository. To use the pre-trained model:

Download the model from this Google Drive link (replace with actual link).
Save it in the root directory as vehicle_color_model.keras.
Usage
Run the application:

bash
Copy code
python main.py
Select a video file or start the webcam to begin detection.

Press q to quit the video processing.

File Structure
graphql
Copy code
vehicle-color-recognition/
│
├── main.py                  # Main GUI and detection logic
├── model_training.ipynb     # Notebook used for training the model
├── requirements.txt         # Python dependencies
├── dataset/                 # Contains the Kaggle dataset
├── haarcascades/            # Haar cascade XML files for detection
│   ├── haarcascade_car.xml
│   ├── haarcascade_fullbody.xml
└── README.md                # Project documentation
Model Training
The training notebook (model_training.ipynb) includes all steps to preprocess data, build, and train the model. It saves the trained model in .keras format. If you wish to retrain the model:

Ensure the dataset is in the dataset/ folder.
Open the notebook and follow the steps.
Limitations
The .keras model file is too large for GitHub; download it via the provided link.
Color recognition may occasionally misclassify due to similarities in vehicle colors or lighting conditions.
Future Enhancements
Improve accuracy by fine-tuning the model on more diverse datasets.
Optimize GUI for better user experience.
Add support for more object detection frameworks like YOLOv5.
