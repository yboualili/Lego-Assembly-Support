from time import time
import numpy as np
import torch
from ultralytics import YOLO
import cv2


class ObjectDetection:
    """
    A class to perform object detection using the YOLOv8 model.
    """

    def __init__(self):
        """
        Initializes the ObjectDetection class.
        """

        # Determine which device to use (CUDA, MPS, or CPU)
        self.device = (
            "cuda"
            if torch.cuda.is_available()  # Use GPU if available
            else "mps"
            if torch.backends.mps.is_available()  # Use MPS (Metal Performance Shaders) on Mac if available
            else "cpu"  # Default to CPU
        )
        print('Using Device:', self.device)

        # Load the YOLO model
        self.model = self.load_model()

        # Set the confidence threshold for predictions
        self.model.conf = 0.3

        # Create a dictionary mapping class indices to class names
        self.CLASS_NAME_DICT = self.model.model.names

        # Generate random colors for each class
        self.colors = {self.CLASS_NAME_DICT[i]: list(np.random.random(size=3) * 256) for i in
                       range(len(self.CLASS_NAME_DICT))}

    def get_class_name_dict(self):
        """
        Returns the class name dictionary of the YOLO model.
        """
        return self.CLASS_NAME_DICT

    def load_model(self):
        """
        Loads the pre-trained YOLO model from a given path.

        Returns:
            model: The loaded YOLO model.
        """
        model = YOLO('yolov8/detect/yolov8n/weights/best.pt')
        print('Model loaded successfully.')
        return model

    def plot_bboxes(self, frame, results, filtered_names):
        """
        Draw bounding boxes and class labels on the frame.

        Args:
            frame (np.ndarray): The input image/frame.
            results (list): The detection results from the YOLO model.
            filtered_names (list): List of class names to filter and display in the output.

        Returns:
            np.ndarray: The frame with bounding boxes and labels drawn.
        """

        # Set the line width for drawing bounding boxes
        lw = max(round(sum(frame.shape) / 2 * 0.003), 2)

        # Extract bounding box coordinates and class ids from the results
        xyxys = results[0].cpu().boxes.xyxy.numpy()
        class_ids = results[0].cpu().boxes.cls.numpy().astype(int)

        # Loop through all the detected boxes and draw them
        for i in range(len(xyxys)):
            xyxy = xyxys[i]  # Bounding box coordinates [xmin, ymin, xmax, ymax]
            pt1 = (int(xyxy[0]), int(xyxy[1]))  # Top-left corner
            pt2 = (int(xyxy[2]), int(xyxy[3]))  # Bottom-right corner
            label = self.CLASS_NAME_DICT[class_ids[i]]  # Get class name from class id

            # If the label is in the filtered_names list, process the box
            if label in filtered_names:
                color = self.colors[label]  # Choose color for the bounding box
                # Choose text color based on background color brightness
                text_color = (255, 255, 255) if sum(color) < 255 * 3 * 0.7 else (0, 0, 0)

                # Draw the bounding box
                cv2.rectangle(frame, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

                # Draw the label background box for the class name
                tf = 1  # Font thickness for the label
                w, h = cv2.getTextSize(label, 0, fontScale=lw / 5, thickness=tf)[0]
                outside = pt1[1] - h >= 3  # Check if label text can fit above the box
                p2 = pt1[0] + w, pt1[1] - h - 3 if outside else pt1[1] + h + 3
                cv2.rectangle(frame, pt1, p2, color, -1, cv2.LINE_AA)  # Filled background

                # Put the class label text on the frame
                cv2.putText(frame,
                            label, (pt1[0], pt1[1] - 2 if outside else pt1[1] + h + 2),
                            0, lw / 5, text_color, thickness=tf, lineType=cv2.LINE_AA)

                # Remove the label from filtered names list once it is plotted
                filtered_names.remove(label)

        return frame

    def predict(self, frame):
        """
        Make a prediction on the given frame using the YOLO model.

        Args:
            frame (np.ndarray): The input image/frame for detection.

        Returns:
            results: The detection results, which include bounding boxes, class ids, and confidences.
        """
        # Perform object detection on the frame with a confidence threshold of 0.6
        result = self.model(frame, conf=0.6)

        return result
