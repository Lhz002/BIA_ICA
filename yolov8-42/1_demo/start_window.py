#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：ultralytics-8.2.77 
@File    ：start_window.py
@IDE     ：PyCharm 
@Author  ：
@Description  ：
@Date    ：
"""
import copy  # Used for image copying
import os  # Used for system path searching
import shutil  # Used for copying
from PySide6.QtGui import *  # GUI components
from PySide6.QtCore import *  # Fonts, margins, and other system variables
from PySide6.QtWidgets import *  # Windows and other small widgets
import threading  # Multithreading
import sys  # System library
import cv2  # OpenCV image processing
import torch  # Deep learning framework
import os.path as osp  # Path searching
import time  # Time computation
from ultralytics import YOLO  # YOLO core algorithm
from PySide6.QtCore import QCoreApplication


os.environ["QT_PLUGIN_PATH"] = r"d:\anaconda\envs\cv\lib\site-packages\PySide6\plugins"
QCoreApplication.addLibraryPath(os.environ["QT_PLUGIN_PATH"])


# Common string constants
WINDOW_TITLE = "Target detection system"  # Title at the top of the system
WELCOME_SENTENCE = (
    "Welcome to the YOLOv8-based cell detection system"  # Welcome sentence
)
ICON_IMAGE = "images/UI/main.png"  # System logo image
IMAGE_LEFT_INIT = (
    "images/UI/up.jpeg"  # Left side image for the detection interface initialization
)
IMAGE_RIGHT_INIT = "images/UI/right.png"  # Right side image for the detection interface initialization


class MainWindow(QTabWidget):
    def __init__(self):
        # Initialize interface
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)  # System window title
        self.resize(1200, 800)  # Initial size of the system
        self.setWindowIcon(QIcon(ICON_IMAGE))  # System logo image
        self.output_size = 480  # Size of the uploaded image and video displayed on the system interface
        self.img2predict = ""  # Path of the image to be predicted
        # self.device = 'cpu'
        self.init_vid_id = "0"  # Camera change
        self.vid_source = int(self.init_vid_id)
        self.cap = cv2.VideoCapture(self.vid_source)
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        self.model_path = "runs/detect/yolov8n/weights/best.pt"  # Specify the location of the model to load
        self.model = self.model_load(weights=self.model_path)
        self.conf_thres = 0.25  # Confidence threshold
        self.iou_thres = 0.45  # IOU threshold for NMS operation
        self.vid_gap = 30  # Interval for saving video frames from the camera
        self.initUI()  # Initialize the graphical interface
        self.reset_vid()  # Reset video parameters to avoid errors during video loading

    # Model initialization
    @torch.no_grad()
    def model_load(self, weights=""):
        """
        Load the model
        """
        model_loaded = YOLO(weights)
        return model_loaded

    def initUI(self):
        """
        Initialize graphical interface
        """
        # ********************* Image detection interface *****************************
        font_title = QFont("KaiTi", 16)
        font_main = QFont("KaiTi", 14)
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("Image Detection Function")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap(IMAGE_LEFT_INIT))
        self.right_img.setPixmap(QPixmap(IMAGE_RIGHT_INIT))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addWidget(self.right_img)
        self.img_num_label = QLabel("Current detection result: Waiting for detection")
        self.img_num_label.setFont(font_main)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("Upload Image")
        det_img_button = QPushButton("Start Detection")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}"
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        det_img_button.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}"
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.img_num_label)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # ********************* Video detection interface *****************************
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        vid_title = QLabel("video detection function")
        vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("camera detection")
        self.mp4_detection_btn = QPushButton("video detection")
        self.vid_stop_btn = QPushButton("stop detection")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}"
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        self.mp4_detection_btn.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}"
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        self.vid_stop_btn.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}"
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)

        self.vid_num_label = QLabel(
            "Current detection result: {}".format("waiting for detection")
        )
        self.vid_num_label.setFont(font_main)
        vid_detection_layout.addWidget(self.vid_num_label)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)
        # ********************* Model selection *****************************
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel(WELCOME_SENTENCE)
        about_title.setFont(QFont("Arial", 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap("images/UI/main.png"))
        self.model_label = QLabel("Current model：{}".format(self.model_path))
        self.model_label.setFont(font_main)
        change_model_button = QPushButton("Switch Model")
        change_model_button.setFont(font_main)
        change_model_button.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}"
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )

        record_button = QPushButton("Chech Record")
        record_button.setFont(font_main)
        record_button.clicked.connect(self.check_record)
        record_button.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: rgb(2,110,180);}"
            "QPushButton{background-color:rgb(48,124,208)}"
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        change_model_button.clicked.connect(self.change_model)
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel()  # todo 更换作者信息
        label_super.setText("BIA Group 7")
        label_super.setFont(QFont("Times New Roman", 16))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addWidget(self.model_label)
        about_layout.addStretch()
        about_layout.addWidget(change_model_button)
        about_layout.addWidget(record_button)
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)
        self.left_img.setAlignment(Qt.AlignCenter)

        self.addTab(about_widget, "Home")
        self.addTab(img_detection_widget, "Picture Detection")
        self.addTab(vid_detection_widget, "Video Detection")

        self.setTabIcon(0, QIcon(ICON_IMAGE))
        self.setTabIcon(1, QIcon(ICON_IMAGE))
        self.setTabIcon(2, QIcon(ICON_IMAGE))

        # ********************* Change video source *****************************

    def upload_img(self):
        """Upload image to detect"""
        fileName, fileType = QFileDialog.getOpenFileName(
            self, "Choose file", "", "*.jpg *.png *.tif *.jpeg"
        )
        if fileName:
            # Check if the user has selected an image, and if so, execute the following operations
            suffix = fileName.split(".")[-1]
            save_path = osp.join(
                "images/tmp", "tmp_upload." + suffix
            )  # Move the image to the "images" directory and rename it
            shutil.copy(fileName, save_path)
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = (
                save_path  # Assign the variable for easy access during prediction
            )
            # Display the image in the UI and initialize the predicted text
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            self.right_img.setPixmap(QPixmap(IMAGE_RIGHT_INIT))
            self.img_num_label.setText("Current detection result: Pending")

    def change_model(self):
        """Switch model and reassign to self.model"""
        fileName, fileType = QFileDialog.getOpenFileName(
            self, "Choose file", "", "*.pt"
        )
        if fileName:
            # If the user selects a pt file, reinitialize the model with the selected pt file
            self.model_path = fileName
            self.model = self.model_load(weights=self.model_path)
            QMessageBox.information(self, "Success", "Model switched successfully!")
            self.model_label.setText("Current model: {}".format(self.model_path))

    # Image Detection
    def detect_img(self):
        """Detect a single image file"""
        output_size = self.output_size
        results = self.model(self.img2predict)  # Load the image and run detection
        result = results[0]  # Get the detection results
        img_array = result.plot()  # Draw detection results on the image
        im0 = img_array
        im_record = copy.deepcopy(im0)
        resize_scale = output_size / im0.shape[0]
        im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
        cv2.imwrite("images/tmp/single_result.jpg", im0)
        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
        time_re = str(time.strftime("result_%Y-%m-%d_%H-%M-%S_%A"))
        cv2.imwrite("record/img/{}.jpg".format(time_re), im_record)
        # Save txt log file (if any)
        # if len(txt_results) > 0:
        #     np.savetxt('record/img/{}.txt'.format(time_re), np.array(txt_results), fmt="%s %s %s %s %s %s", delimiter="\n")
        # Get the count of each detected class and display on the GUI
        result_names = result.names
        result_nums = [0 for i in range(0, len(result_names))]
        cls_ids = list(result.boxes.cls.cpu().numpy())
        for cls_id in cls_ids:
            result_nums[int(cls_id)] = result_nums[int(cls_id)] + 1
        result_info = ""
        for idx_cls, cls_num in enumerate(result_nums):
            # Check if the count is greater than 0, then add the result to the info string
            if cls_num > 0:
                result_info = result_info + "{}:{}\n".format(
                    result_names[idx_cls], cls_num
                )
        self.img_num_label.setText("Current detection result:\n {}".format(result_info))
        QMessageBox.information(self, "Detection Successful", "Log has been saved!")

    def open_cam(self):
        """Open webcam for real-time detection"""
        self.webcam_detection_btn.setEnabled(
            False
        )  # Disable the webcam button to prevent accidental clicks
        self.mp4_detection_btn.setEnabled(
            False
        )  # Disable the mp4 button to prevent accidental clicks
        self.vid_stop_btn.setEnabled(
            True
        )  # Enable the stop button to allow the user to stop the detection
        self.vid_source = int(self.init_vid_id)  # Reinitialize the webcam
        self.webcam = True  # Set webcam mode to True
        self.cap = cv2.VideoCapture(self.vid_source)  # Initialize the webcam object
        th = threading.Thread(
            target=self.detect_vid
        )  # Initialize the video detection thread
        th.start()  # Start the detection thread

    def open_mp4(self):
        """Open an mp4 file for detection"""
        fileName, fileType = QFileDialog.getOpenFileName(
            self, "Choose file", "", "*.mp4 *.avi"
        )
        if fileName:
            # Similar to the open_cam method, except here the video source is set to the mp4 file instead of the webcam
            self.webcam_detection_btn.setEnabled(
                False
            )  # Disable webcam detection button
            self.mp4_detection_btn.setEnabled(False)  # Disable mp4 detection button
            self.vid_source = fileName
            self.webcam = False  # Set webcam mode to False (we are using mp4 file now)
            self.cap = cv2.VideoCapture(
                self.vid_source
            )  # Open the video file using OpenCV
            th = threading.Thread(
                target=self.detect_vid
            )  # Initialize the video detection thread
            th.start()  # Start the thread for detection

    # Video Detection Main Function
    def detect_vid(self):
        """Detect video files (both mp4 and webcam video sources)"""
        vid_i = 0
        while self.cap.isOpened():
            # Read a frame from the video
            success, frame = self.cap.read()
            if success:
                # Perform YOLOv8 inference on the frame
                results = self.model(frame)
                result = results[0]
                img_array = result.plot()
                # Detect, display, and save the resulting image
                im0 = img_array
                im_record = copy.deepcopy(im0)
                resize_scale = self.output_size / im0.shape[0]
                im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/tmp/single_result_vid.jpg", im0)
                self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                time_re = str(time.strftime("result_%Y-%m-%d_%H-%M-%S_%A"))
                if vid_i % self.vid_gap == 0:
                    cv2.imwrite("record/vid/{}.jpg".format(time_re), im_record)
                # Save txt log file (if any)
                # if len(txt_results) > 0:
                #     np.savetxt('record/img/{}.txt'.format(time_re), np.array(txt_results), fmt="%s %s %s %s %s %s", delimiter="\n")
                # Count the number of occurrences of each class and display them on the UI
                result_names = result.names
                result_nums = [0 for i in range(0, len(result_names))]
                cls_ids = list(result.boxes.cls.cpu().numpy())
                for cls_id in cls_ids:
                    result_nums[int(cls_id)] = result_nums[int(cls_id)] + 1
                result_info = ""
                for idx_cls, cls_num in enumerate(result_nums):
                    if cls_num > 0:
                        result_info = result_info + "{}:{}\n".format(
                            result_names[idx_cls], cls_num
                        )
                self.vid_num_label.setText(
                    "Current detection result:\n {}".format(result_info)
                )
                vid_i += 1
            if cv2.waitKey(1) & self.stopEvent.is_set() == True:
                # Close and release the video resources
                self.stopEvent.clear()
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
                if self.cap is not None:
                    self.cap.release()
                    cv2.destroyAllWindows()
                self.reset_vid()
                break

    # Reset the webcam for new detection
    def reset_vid(self):
        """Reset webcam content"""
        self.webcam_detection_btn.setEnabled(True)  # Enable webcam detection button
        self.mp4_detection_btn.setEnabled(True)  # Enable mp4 file detection button
        self.vid_img.setPixmap(
            QPixmap(IMAGE_LEFT_INIT)
        )  # Reset the initial image in the video detection UI
        self.vid_source = int(self.init_vid_id)  # Reset the video source
        self.webcam = True  # Set webcam mode back to True
        self.vid_num_label.setText(
            "Current detection result: {}".format("Waiting for detection")
        )  # Reset detection status text

    def close_vid(self):
        """Close webcam"""
        self.stopEvent.set()
        self.reset_vid()

    def check_record(self):
        """Open historical record folder"""
        os.startfile(osp.join(os.path.abspath(os.path.dirname(__file__)), "record"))

    def closeEvent(self, event):
        """Handle user exit event"""
        reply = QMessageBox.question(
            self,
            "Quit",
            "Are you sure you want to quit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                # Ensure webcam resources are released after quitting to avoid keeping the camera active
                if self.cap is not None:
                    self.cap.release()
                    print("Webcam released")
            except:
                pass
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
