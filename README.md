# Car-color-detection


Certainly! Here’s an elaborated version of each section of the project report to ensure a comprehensive and professional document:  

---

### **Project Report: Car Color Detection Model**  

#### **1. Abstract**  
This project introduces a machine learning-based car color detection system to enhance traffic monitoring capabilities. The system classifies car colors in real-time and counts the number of cars at traffic signals. Unique visual markers, such as red rectangles for blue cars and blue rectangles for other cars, ensure clarity. Additionally, the model detects and counts people present at the signal. A user-friendly graphical interface (GUI) is developed to preview the input and processed output.  

#### **2. Introduction**  
- **Problem Statement**:  
   Traffic management is a critical challenge in urban areas. Identifying vehicle attributes like color can assist in traffic analysis, incident management, and law enforcement. Similarly, tracking pedestrian presence enhances safety. This project bridges the gap by providing an automated solution for detecting car colors and pedestrians in traffic.  

- **Objectives**:  
   - Develop a robust car color classification model using deep learning.  
   - Detect cars and pedestrians in video feeds using pre-trained object detection models.  
   - Create a GUI for intuitive use and real-time feedback.  

- **Applications**:  
   - Traffic monitoring and incident detection.  
   - Pedestrian safety systems at traffic signals.  
   - Data collection for urban planning.  

#### **3. Literature Review**  
- **Object Detection in Traffic Monitoring**:  
   Traditional systems rely on sensors or manual observation, which are costly and prone to errors. Computer vision offers a scalable and accurate alternative.  
   
- **Haar Cascades**:  
   Haar cascades are widely used for object detection due to their efficiency and simplicity. Pre-trained Haar cascades for vehicles and pedestrians are employed in this project.  

- **Deep Learning for Color Classification**:  
   CNNs (Convolutional Neural Networks) are state-of-the-art models for image classification tasks. Their ability to learn spatial hierarchies makes them ideal for tasks like car color detection.  

#### **4. Methodology**  

- **Data Collection**:  
   The dataset is sourced from Kaggle, comprising labeled car images of various colors like black, blue, red, etc. Images are divided into training, validation, and testing subsets for model development and evaluation.  

- **Image Preprocessing**:  
   - **Resizing**: All images are resized to \(224 \times 224\) to match the input dimensions of the CNN.  
   - **Normalization**: Pixel values are scaled to [0, 1] for faster convergence during training.  
   - **Augmentation**: Techniques like rotation, flipping, and brightness adjustments enhance model generalization.  

- **Model Training**:  
   - **Architecture**: The CNN uses convolutional, pooling, and fully connected layers to extract features and classify images.  
   - **Loss Function**: Categorical cross-entropy is used to optimize predictions for multi-class classification.  
   - **Evaluation Metrics**: Accuracy, precision, recall, and F1 score.  

- **Car and Person Detection**:  
   - **Haar Cascades**: Pre-trained cascades (`haarcascade_car.xml` and `haarcascade_fullbody.xml`) identify cars and people in video frames.  
   - **Detection Flow**:  
      1. Convert frames to grayscale for detection.  
      2. Extract regions of interest (ROI) for cars and classify their colors.  
      3. Highlight detected cars and people using bounding rectangles.  

- **GUI Development**:  
   - **Design**: The GUI, built using Tkinter, allows users to select video files or access real-time webcam feeds.  
   - **Interactivity**: Outputs are displayed with annotated bounding boxes and labels.  

#### **5. Implementation**  

- **Key Modules**:  
   1. **Color Classification**: Predicts car color from detected ROIs using the trained model.  
   2. **Object Detection**: Uses Haar cascades for car and person identification.  
   3. **GUI**: Facilitates user interaction with options to upload videos or start live detection.  

- **Code Snippets**:  

   **Image Preprocessing for Model Input**:  
   ```python
   def prepare_image(image):
       image = cv2.resize(image, (224, 224))
       image = image / 255.0
       image = np.expand_dims(image, axis=0)
       return image
   ```  

   **GUI for File Selection and Webcam Feed**:  
   ```python
   def open_file():
       file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
       if file_path:
           process_feed(file_path)
   ```  

   **Detection and Annotation**:  
   ```python
   for (x, y, w, h) in cars:
       car_roi = frame[y:y+h, x:x+w]
       prediction = model.predict(prepare_image(car_roi))
       car_color = class_labels[np.argmax(prediction)]
       rectangle_color = (0, 0, 255) if car_color == "blue" else (255, 0, 0)
       cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 2)
   ```  

#### **6. Results**  

- **Performance Metrics**:  
   - Training accuracy: \( \sim 90\% \)  
   - Validation accuracy: \( \sim 88\% \)  

- **Screenshots**:  
   Include annotated screenshots of detected cars and people with bounding rectangles and color labels.  

#### **7. Challenges and Limitations**  

- **Challenges**:  
   - Handling varying lighting conditions and occlusions in video feeds.  
   - Balancing model accuracy and real-time performance.  

- **Limitations**:  
   - Haar cascades can sometimes miss objects in complex backgrounds.  
   - Color prediction accuracy might drop for low-resolution ROIs.  

#### **8. Conclusion**  
This project successfully demonstrates an automated car color detection and pedestrian counting system. The integration of a machine learning model with real-time object detection provides an efficient solution for traffic monitoring. Future enhancements could include multi-camera setups and integration with traffic management systems.  

#### **9. References**  

1. Kaggle Vehicle Color Recognition Dataset.  
2. OpenCV Documentation for Haar Cascades.  
3. Keras Documentation for Deep Learning Models.  

---

Let me know if you’d like help formatting this report or adding diagrams/screenshots! 😊



## Features

- **Car Color Detection:** Recognizes the color of vehicles from a predefined set of colors (e.g., yellow, red, blue, etc.).
- **Dynamic Annotations:**
  - **Blue Rectangle:** Drawn around vehicles of colors other than "blue."
  - **Red Rectangle:** Drawn around vehicles detected as "blue."
  - **Green Rectangle:** Drawn around detected people, including their count.
- **Haar Cascade Integration:** Detects vehicles and people using pre-trained Haar cascades.
- **GUI Interface:** Built using Tkinter to allow users to process video files or live webcam feeds.

## Dataset

The project uses the [VCOR - Vehicle Color Recognition Dataset](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset) from Kaggle, which provides labeled images of vehicles in various colors. 

## Requirements

- Python 3.8 or higher
- OpenCV
- TensorFlow/Keras
- NumPy
- Tkinter (default with Python)
- Pre-trained Haar cascades (`haarcascade_car.xml`, `haarcascade_fullbody.xml`)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/vehicle-color-recognition.git
   cd vehicle-color-recognition
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download and extract the dataset from Kaggle:

   - [VCOR - Vehicle Color Recognition Dataset](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset)

   Place the dataset in a folder named `dataset` within the project directory.

4. Place the `.keras` model file:

   Due to its large size, the trained `.keras` model is not included in this repository. To use the pre-trained model:
   - Download the model from this [[Google Drive link](https://drive.google.com/drive/folders/1qsjD8CMuT5IU3eQ2XtIxaIE2TESI6iet?usp=drive_link)](#) .
   - Save it in the root directory as `vehicle_color_model.keras`.

## Usage

1. Run the application:

   ```bash
   python main.py
   ```

2. Select a video file or start the webcam to begin detection.

3. Press `q` to quit the video processing.

## File Structure

```
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
```

## Model Training

The training notebook (`model_training.ipynb`) includes all steps to preprocess data, build, and train the model. It saves the trained model in `.keras` format. If you wish to retrain the model:
1. Ensure the dataset is in the `dataset/` folder.
2. Open the notebook and follow the steps.

## Limitations

- The `.keras` model file is too large for GitHub; download it via the provided link.
- Color recognition may occasionally misclassify due to similarities in vehicle colors or lighting conditions.

## Future Enhancements

- Improve accuracy by fine-tuning the model on more diverse datasets.
- Optimize GUI for better user experience.
- Add support for more object detection frameworks like YOLOv5.

