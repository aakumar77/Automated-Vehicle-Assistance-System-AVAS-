
# YOLOv12-Based Lane and Object Detection System

This project combines **YOLOv12 object detection** with **lane detection**, real-time video processing, and distance estimation to simulate a smart driving assistant system. It processes road scenes from video input, detects key road objects, highlights lanes, and provides distance estimates for enhanced situational awareness.

---

## ✅ Features

- Real-time object detection using YOLOv12:
  - Detects classes such as `car`, `bike`, `person`, `traffic light`, `traffic sign`.
- Distance estimation of detected objects using bounding box width.
- Lane detection using:
  - Canny edge detection.
  - Hough Transform.
  - Region of Interest (ROI) masking.
- Real-time video input processing with OpenCV.
- CUDA acceleration for fast inference.

---

## 📁 Files Overview

| File                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `Object Detection.py` | Python script that handles video input, object & lane detection, and distance estimation. |
| `YOLOv12_VS.ipynb`    | Jupyter notebook used for testing and validating YOLOv12 model performance. |

---

## 🛠 Requirements

- Python 3.8+
- Libraries:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
- Optional: CUDA-enabled GPU for optimal performance

```bash
pip install ultralytics opencv-python numpy
```

---

## 🚀 Getting Started

1. **Clone the repository**
   - Use the following commands:
     ```bash
     git clone https://github.com/aakumar77/yolov12-object-lane-detection.git
     cd yolov12-object-lane-detection
     ```

2. **Prepare your assets**
   - Add your YOLOv12 weights:
     - Place the weights at `weights/yolo12s.pt`
   - Add the video input:
     - Place your video at `video/Test_vid - Made with Clipchamp.mp4`

3. **Run the detection script**
   - Execute the script using:
     ```bash
     python Object\ Detection.py
     ```

4. **Quit the preview**
   - Press `q` while the video window is active to close it.

---

## 🧠 How It Works

- **YOLOv12 Object Detection**
  - Loads YOLOv12 model using Ultralytics API.
  - Detects multiple classes frame-by-frame.
  - Filters detections based on confidence score.
- **Lane Detection Pipeline**
  - Converts frame to grayscale.
  - Applies Gaussian blur and Canny edge detection.
  - Uses ROI masking to focus on lane region.
  - Applies Hough Transform to detect lines.
  - Classifies lines as left or right lanes.
- **Distance Estimation**
  - Calculates approximate distance using bounding box width and focal length formula.
- **Frame Rendering**
  - Draws bounding boxes and distance labels.
  - Overlays detected lane lines.
  - Displays the processed frame in real-time.

---

## 📊 Model Evaluation (`YOLOv12_VS.ipynb`)

- Evaluates YOLOv12 model on custom dataset.
- Provides:
  - Detection confidence visualizations.
  - Precision and recall checks.
  - Comparison between different YOLOv12 variants (`yolov12s`, `yolov12m`, etc.).
- Uses Ultralytics `model.val()` and visualization tools.

---

## 📸 Sample Output

- Example outputs will be added soon.
  - Sample frame with bounding boxes and distance labels.
  - Lane overlays on detected road segments.

---

## 📌 Notes

- You can improve accuracy by:
  - Adjusting lane ROI based on video resolution.
  - Calibrating distance estimation constants.
- Additional module ideas:
  - Traffic sign detection using EasyOCR.
  - Vehicle tracking using SORT.
  - Pothole detection with CNNs or segmentation models.

---

## 📄 License

- This project is for **research and educational purposes only**.
- No commercial use is permitted without permission.

---

## 👨‍💻 Author

- **Name**: Aayush
- **GitHub**: [Link](https://github.com/aakumar77)
- **LinkedIn**: [Link](https://linkedin.com/in/aayush-kumar-0811212a3)
