#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-8.2.77
@File    ：start_tracing.py
@IDE     ：PyCharm
@Author  ：
@Description  ：Load trained model and predict image with UI
@Date    ：2024/11/15 15:15
'''
import sys
import threading
import cv2
import copy
import torch
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QFileDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,  # 新增导入
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, QObject

from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

# Temporary workaround for OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Tracker type mapping
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


# Tracker initialization and update callback functions
def on_predict_start(predictor: object, persist: bool = False) -> None:
    """
    Initialize trackers for object tracking during prediction.
    """
    if hasattr(predictor, "trackers") and persist:
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))

    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(
            f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'"
        )

    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # only need one tracker for other modes.
            break
    predictor.trackers = trackers
    predictor.vid_path = [
        None
    ] * predictor.dataset.bs  # for determining when to reset tracker on new video


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """
    Postprocess detected boxes and update with object tracking.
    """
    path, im0s = predictor.batch[:2]

    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"
    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (
            (predictor.results[i].obb if is_obb else predictor.results[i].boxes)
            .cpu()
            .numpy()
        )
        if len(det) == 0:
            continue
        tracks = tracker.update(det, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]

        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)


def register_tracker(model: object, persist: bool) -> None:
    """
    Register tracking callbacks to the model for object tracking during prediction.
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback(
        "on_predict_postprocess_end",
        partial(on_predict_postprocess_end, persist=persist),
    )


class DetectionSignals(QObject):
    update_image = pyqtSignal(str)
    update_info = pyqtSignal(str)
    no_cell_detected = pyqtSignal()  # 新增信号


class DetectionClass:
    def __init__(
        self, model_path, tracker_config, output_size=640, vid_gap=30, signals=None
    ):
        self.model_path = model_path
        self.model = None
        self.tracker_config = tracker_config
        self.output_size = output_size
        self.vid_gap = vid_gap
        self.cap = None
        self.stopEvent = threading.Event()
        self.trackers = []
        self.vid_path = []
        self.signals = signals  # Signal object for communication with the main thread
        self.frame_count = 0  # 新增
        self.cells_detected = False  # 新增
        # Initialize tracker
        self.initialize_tracker()
        # Load model
        self.load_model()

    def load_model(self):
        """
        Load the YOLO model.
        """
        if self.model_path:
            self.model = YOLO(self.model_path)
            print(f"Model loaded: {self.model_path}")
        else:
            raise ValueError("Model path is not specified.")

    def initialize_tracker(self):
        """
        Initialize the tracker using the provided tracker configuration.
        """
        tracker = check_yaml(self.tracker_config)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        if cfg.tracker_type not in TRACKER_MAP:
            raise ValueError(f"Unsupported tracker type: {cfg.tracker_type}")

        # Initialize tracker(s) based on batch size or video mode
        self.trackers = [TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)]
        self.vid_path = [None]

    def detect_vid(self):
        """Detect video frames, including MP4 files and camera streams."""
        vid_i = 0
        while self.cap.isOpened():
            try:
                # Read a frame from the video
                success, frame = self.cap.read()
                if success:
                    self.frame_count +=1  # 新增

                    # Run YOLOv8 inference on the frame
                    results = self.model(frame)
                    result = results[0]

                    # Get detection results
                    det = result.boxes.cpu().numpy()

                    if self.frame_count <=3:
                        if len(det) >0:
                            self.cells_detected = True

                    if self.frame_count ==3:
                        if not self.cells_detected:
                            # 未检测到细胞，发射信号
                            if self.signals:
                                self.signals.no_cell_detected.emit()
                            # 停止检测
                            break

                    if self.frame_count >3 and len(det) ==0:
                        # 继续处理下一帧
                        pass

                    if len(det) > 0:
                        # Extract detection boxes (x1, y1, x2, y2, confidence, class)
                        boxes = det[:, :6]

                        # Update tracker
                        tracks = self.trackers[0].update(boxes, frame)

                        # Visualize tracking results
                        for track in tracks:
                            x1, y1, x2, y2, track_id = track[:5]
                            cls_id = int(track[5])
                            cls_name = result.names[cls_id]
                            color = (
                                0,
                                255,
                                0,
                            )  # Can generate different colors based on track_id
                            cv2.rectangle(
                                frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
                            )
                            cv2.putText(
                                frame,
                                f"ID: {track_id} {cls_name}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                            )

                        # Only visualize tracking results and save the resulting image
                        im_record = copy.deepcopy(frame)
                        resize_scale = self.output_size / frame.shape[0]
                        im0 = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                        cv2.imwrite("images/tmp/single_result_vid.jpg", im0)

                        if self.signals:
                            # Emit signal to update UI image
                            self.signals.update_image.emit(
                                "images/tmp/single_result_vid.jpg"
                            )

                        time_re = str(time.strftime("result_%Y-%m-%d_%H-%M-%S_%A"))
                        if vid_i % self.vid_gap == 0:
                            cv2.imwrite(f"record/vid/{time_re}.jpg", im_record)

                        # Count the number of each category and display
                        result_names = result.names
                        result_nums = [0 for _ in range(len(result_names))]
                        cls_ids = list(result.boxes.cls.cpu().numpy())
                        for cls_id in cls_ids:
                            result_nums[int(cls_id)] += 1
                        result_info = ""
                        for idx_cls, cls_num in enumerate(result_nums):
                            if cls_num > 0:
                                result_info += f"{result_names[idx_cls]}: {cls_num}\n"
                        if self.signals:
                            self.signals.update_info.emit(
                                f"Detection Results:\n{result_info}"
                            )
                        vid_i += 1
                    else:
                        # No detection in frame beyond first ten frames
                        pass

                    # Check if need to stop
                    if cv2.waitKey(1) & 0xFF == ord("q") or self.stopEvent.is_set():
                        # Close and release video resources
                        self.stopEvent.clear()
                        if self.cap is not None:
                            self.cap.release()
                            cv2.destroyAllWindows()
                        self.reset_vid()
                        break

                else:
                    # Video ended
                    break

            except Exception as e:
                print(f"Error in detection thread: {e}")
                break

        # After loop ends, check if frames were less than 10 and no cells detected
        if self.frame_count <10 and not self.cells_detected:
            if self.signals:
                self.signals.no_cell_detected.emit()

        # Release resources after video processing ends
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Detection thread terminated.")

    def stop_detection(self):
        """
        Stop the detection thread.
        """
        self.stopEvent.set()

    def reset_vid(self):
        """
        Reset video-related parameters or state.
        """
        # Implement video reset logic, e.g., clear trackers, reset counters, etc.
        self.trackers = []
        self.vid_path = []
        self.initialize_tracker()
        print("Video has been reset.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Video Detection and Tracking")
        self.setGeometry(100, 100, 1200, 800)  # Increased window size
        self.setFixedSize(1200, 800)  # Set fixed window size

        # Central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Top layout for buttons and labels
        self.top_layout = QHBoxLayout()
        self.main_layout.addLayout(self.top_layout)

        # Select Model Button and Label
        self.select_model_btn = QPushButton("Select Model", self)
        self.select_model_btn.clicked.connect(self.select_model)
        self.top_layout.addWidget(self.select_model_btn)

        self.model_label = QLabel("No model selected", self)
        self.top_layout.addWidget(self.model_label)

        # Select Tracker Button and Label
        self.select_tracker_btn = QPushButton("Select Tracker", self)
        self.select_tracker_btn.clicked.connect(self.select_tracker)
        self.top_layout.addWidget(self.select_tracker_btn)

        self.tracker_label = QLabel("No tracker selected", self)
        self.top_layout.addWidget(self.tracker_label)

        # Open Camera Button
        self.webcam_detection_btn = QPushButton("Open Camera", self)
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.webcam_detection_btn.setEnabled(False)  # Initially disabled
        self.top_layout.addWidget(self.webcam_detection_btn)

        # Open MP4 File Button
        self.mp4_detection_btn = QPushButton("Open MP4 File", self)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.mp4_detection_btn.setEnabled(False)  # Initially disabled
        self.top_layout.addWidget(self.mp4_detection_btn)

        # Stop Detection Button
        self.vid_stop_btn = QPushButton("Stop Detection", self)
        self.vid_stop_btn.clicked.connect(self.stop_detection)
        self.vid_stop_btn.setEnabled(False)
        self.top_layout.addWidget(self.vid_stop_btn)

        # Video Display Label
        self.vid_img = QLabel(self)
        self.vid_img.setFixedSize(1100, 600)  # Adjusted to fit window
        self.vid_img.setStyleSheet("background-color: black;")
        self.vid_img.setScaledContents(True)
        self.main_layout.addWidget(self.vid_img)

        # Detection Info Label
        self.vid_num_label = QLabel(self)
        self.vid_num_label.setFixedHeight(40)
        self.main_layout.addWidget(self.vid_num_label)

        self.init_vid_id = "0"  # Camera ID, usually 0
        self.output_size = 640
        self.vid_gap = 30
        self.detector = None  # Initialize detector instance
        self.cap = None
        self.stopEvent = threading.Event()

        # Create signal object
        self.signals = DetectionSignals()
        self.signals.update_image.connect(self.update_image)
        self.signals.update_info.connect(self.update_info)
        self.signals.no_cell_detected.connect(self.show_no_cell_detected)  # 连接新信号

        self.selected_model_path = None  # Store selected model path
        self.selected_tracker_path = None  # Store selected tracker config path

        # 新增标志位
        self.no_cell_popup_shown = False

    def select_model(self):
        """Select a model file"""
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "*.pt")
        if fileName:
            self.selected_model_path = fileName
            model_file_name = os.path.basename(fileName)
            self.model_label.setText(f"Model Selected: {model_file_name}")
            # Enable detection buttons only if tracker is also selected
            if self.selected_tracker_path:
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
            print(f"Model selected: {fileName}")

    def select_tracker(self):
        """Select a tracker configuration file"""
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Select Tracker Configuration File", "", "*.yaml"
        )
        if fileName:
            self.selected_tracker_path = fileName
            tracker_file_name = os.path.basename(fileName)
            self.tracker_label.setText(f"Tracker Selected: {tracker_file_name}")
            # Enable detection buttons only if model is also selected
            if self.selected_model_path:
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
            print(f"Tracker selected: {fileName}")

    def update_image(self, image_path):
        """Update image display via signal"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.vid_img.setPixmap(pixmap)
        else:
            self.vid_img.setText("No Image Available")

    def update_info(self, info):
        """Update detection info via signal"""
        self.vid_num_label.setText(info)

    def show_no_cell_detected(self):
        """显示未检测到细胞的警告弹窗，并停止检测"""
        if not self.no_cell_popup_shown:
            QMessageBox.warning(self, "No Cell Detected", "No cell detected, Please check the video")
            self.no_cell_popup_shown = True
            self.stop_detection()

    def open_cam(self):
        """Open camera for detection"""
        if not self.selected_model_path:
            self.vid_num_label.setText("Please select a model file first.")
            return
        if not self.selected_tracker_path:
            self.vid_num_label.setText(
                "Please select a tracker configuration file first."
            )
            return

        self.webcam_detection_btn.setEnabled(
            False
        )  # Disable to prevent multiple triggers
        self.mp4_detection_btn.setEnabled(False)  # Disable to prevent multiple triggers
        self.vid_stop_btn.setEnabled(True)  # Enable stop button

        try:
            self.vid_source = int(self.init_vid_id)  # Initialize camera
        except ValueError:
            self.vid_source = 0  # Default to camera 0 if conversion fails

        self.cap = cv2.VideoCapture(self.vid_source)  # Initialize video capture object

        # Instantiate DetectionClass
        self.detector = DetectionClass(
            model_path=self.selected_model_path,
            tracker_config=self.selected_tracker_path,
            output_size=self.output_size,
            vid_gap=self.vid_gap,
            signals=self.signals,
        )
        self.detector.cap = self.cap

        # Start detection thread
        th = threading.Thread(
            target=self.detector.detect_vid, name="DetectionThread", daemon=True
        )
        th.start()
        print("Camera detection thread started.")

    def open_mp4(self):
        """Open MP4 file for detection"""
        if not self.selected_model_path:
            self.vid_num_label.setText("Please select a model file first.")
            return
        if not self.selected_tracker_path:
            self.vid_num_label.setText(
                "Please select a tracker configuration file first."
            )
            return

        fileName, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "*.mp4 *.avi"
        )
        if fileName:
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            self.vid_stop_btn.setEnabled(True)  # Enable stop button

            self.vid_source = fileName
            self.cap = cv2.VideoCapture(self.vid_source)

            # Instantiate DetectionClass
            self.detector = DetectionClass(
                model_path=self.selected_model_path,
                tracker_config=self.selected_tracker_path,  # Use selected tracker config
                output_size=self.output_size,
                vid_gap=self.vid_gap,
                signals=self.signals,
            )
            self.detector.cap = self.cap

            # Start detection thread
            th = threading.Thread(
                target=self.detector.detect_vid, name="DetectionThread", daemon=True
            )
            th.start()
            print("MP4 detection thread started.")

    def stop_detection(self):
        """Stop the detection thread and release resources"""
        if self.detector:
            self.detector.stop_detection()
            self.detector = None
            print("Detection has been stopped.")

        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_stop_btn.setEnabled(False)
        self.model_label.setText("No model selected")
        self.tracker_label.setText("No tracker selected")
        self.vid_num_label.setText("Detection stopped.")
        self.no_cell_popup_shown = False  # 重置标志位

    def reset_vid(self):
        """Reset video-related parameters or state"""
        if self.detector:
            self.detector.reset_vid()
            print("Video reset.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
