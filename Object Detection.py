import cv2
import numpy as np
import time
from ultralytics import YOLO


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def draw_lane_lines(img, left_line, right_line, color=[0, 255, 0]):
    line_img = np.zeros_like(img)
    if all(left_line) and all(right_line):
        poly_pts = np.array([
            [(left_line[0], left_line[1]),
             (left_line[2], left_line[3]),
             (right_line[2], right_line[3]),
             (right_line[0], right_line[1])]
        ], dtype=np.int32)
        cv2.fillPoly(line_img, poly_pts, color)
    return cv2.addWeighted(img, 0.8, line_img, 0.5, 0.0)


def pipeline(image):
    height, width = image.shape[:2]
    roi_vertices = [(0, height), (width // 2, height // 2), (width, height)]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped, 6, np.pi / 60, 160, minLineLength=40, maxLineGap=25)
    left_x, left_y, right_x, right_y = [], [], [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if abs(slope) < 0.5:
                continue
            (left_x if slope <= 0 else right_x).extend([x1, x2])
            (left_y if slope <= 0 else right_y).extend([y1, y2])

    min_y, max_y = height * 3 // 5, height
    left_line = [0, max_y, 0, min_y] if not left_x else [int(np.poly1d(np.polyfit(left_y, left_x, 1))(max_y)), max_y,
                                                         int(np.poly1d(np.polyfit(left_y, left_x, 1))(min_y)), min_y]
    right_line = [width, max_y, width, min_y] if not right_x else [
        int(np.poly1d(np.polyfit(right_y, right_x, 1))(max_y)), max_y,
        int(np.poly1d(np.polyfit(right_y, right_x, 1))(min_y)), min_y]

    return draw_lane_lines(image, left_line, right_line)


def estimate_distance(bbox_width, focal_length=1000, known_width=2.0):
    return (known_width * focal_length) / bbox_width if bbox_width > 0 else float('inf')


def process_video():
    # Replace 'weights/best.pt' with the path to your custom-trained model
    model = YOLO('weights/yolo12s.pt').to('cuda')

    cap = cv2.VideoCapture('video/Test_vid - Made with Clipchamp.mp4')
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    target_fps = 30
    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (1280, 720))
        lane_frame = pipeline(resized_frame)
        results = model(resized_frame)

        # Iterate over results from the model
        for result in results:  # Iterate over the results (each is a Result object)
            for box in result.boxes:  # Each 'box' contains a bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
                conf, cls = float(box.conf[0]), int(box.cls[0])  # Extract confidence and class id
                class_name = model.names.get(cls, "unknown")  # Get class name from class id

                if conf >= 0.1 and class_name in {"car", "bike", "person", "traffic light", "traffic sign"}:
                    # Draw bounding box
                    cv2.rectangle(lane_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    # Display label
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(lane_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Estimate distance
                    distance = estimate_distance(x2 - x1)
                    cv2.putText(lane_frame, f'Distance: {distance:.2f}m', (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2)

        # Display the processed frame
        cv2.imshow('Lane and Object Detection', lane_frame)

        # Frame rate control
        curr_time = time.time()
        elapsed = curr_time - prev_time
        if elapsed < 1 / target_fps:
            time.sleep((1 / target_fps) - elapsed)
        prev_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the video processing function
process_video()
