"""
Parameter input tab for the application
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QLineEdit, QPushButton, QFileDialog, QComboBox,
                                QTextEdit, QGroupBox, QGridLayout, QMessageBox,
                                QDialog, QDialogButtonBox, QSpinBox, QCheckBox)
from PySide6.QtCore import Signal
import os
import sys
import numpy as np
from pathlib import Path

# Import video extraction function and default configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from video2image import extract_frames
except ImportError:
    extract_frames = None

try:
    import default_cfg
    DEFAULT_DOWNSAMPLE_RATIO = default_cfg.downsample_ratio
    DEFAULT_ENABLE_BA = default_cfg.enable_ba
except ImportError:
    DEFAULT_DOWNSAMPLE_RATIO = 30  # Fallback default
    DEFAULT_ENABLE_BA = False  # Fallback default


class ParameterTab(QWidget):
    parameters_ready = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Image directory selection
        dir_group = QGroupBox("Image Directory / Video File")
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Select directory containing images or video file...")
        dir_browse_btn = QPushButton("Browse...")
        dir_browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(dir_browse_btn)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # ArUco parameters
        aruco_group = QGroupBox("ArUco Marker Parameters")
        aruco_layout = QGridLayout()
        
        # ArUco size selection
        aruco_layout.addWidget(QLabel("ArUco Size:"), 0, 0)
        self.aruco_size_combo = QComboBox()
        self.aruco_size_combo.addItems(["4x4", "5x5", "6x6"])
        aruco_layout.addWidget(self.aruco_size_combo, 0, 1)
        
        # Physical length
        aruco_layout.addWidget(QLabel("Physical Length (meters):"), 1, 0)
        self.physical_length_edit = QLineEdit()
        self.physical_length_edit.setPlaceholderText("e.g., 0.05")
        aruco_layout.addWidget(self.physical_length_edit, 1, 1)
        
        aruco_group.setLayout(aruco_layout)
        layout.addWidget(aruco_group)
        
        # Camera intrinsic (optional)
        camera_group = QGroupBox("Camera Intrinsic (Optional)")
        camera_layout = QGridLayout()
        
        camera_layout.addWidget(QLabel("fx:"), 0, 0)
        self.fx_edit = QLineEdit()
        self.fx_edit.setPlaceholderText("e.g., 800")
        camera_layout.addWidget(self.fx_edit, 0, 1)
        
        camera_layout.addWidget(QLabel("fy:"), 0, 2)
        self.fy_edit = QLineEdit()
        self.fy_edit.setPlaceholderText("e.g., 800")
        camera_layout.addWidget(self.fy_edit, 0, 3)
        
        camera_layout.addWidget(QLabel("cx:"), 1, 0)
        self.cx_edit = QLineEdit()
        self.cx_edit.setPlaceholderText("e.g., 320")
        camera_layout.addWidget(self.cx_edit, 1, 1)
        
        camera_layout.addWidget(QLabel("cy:"), 1, 2)
        self.cy_edit = QLineEdit()
        self.cy_edit.setPlaceholderText("e.g., 240")
        camera_layout.addWidget(self.cy_edit, 1, 3)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # JSON file path (optional, for continuing labeling)
        json_group = QGroupBox("Previous Labeling File (Optional)")
        json_layout = QHBoxLayout()
        self.json_edit = QLineEdit()
        self.json_edit.setPlaceholderText("Select JSON file to continue labeling...")
        json_browse_btn = QPushButton("Browse...")
        json_browse_btn.clicked.connect(self.browse_json)
        json_layout.addWidget(self.json_edit)
        json_layout.addWidget(json_browse_btn)
        json_group.setLayout(json_layout)
        layout.addWidget(json_group)
        
        # Start button and Enable BA checkbox
        start_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        start_layout.addWidget(self.start_btn)
        
        self.enable_ba_checkbox = QCheckBox("Enable BA")
        self.enable_ba_checkbox.setChecked(DEFAULT_ENABLE_BA)
        start_layout.addWidget(self.enable_ba_checkbox)
        
        layout.addLayout(start_layout)
        
        # Camera Intrinsic Optimal (display after bundle adjustment)
        camera_optimal_group = QGroupBox("Camera Intrinsic Optimal (After Bundle Adjustment)")
        camera_optimal_layout = QVBoxLayout()
        self.camera_optimal_label = QLabel("Not available yet")
        self.camera_optimal_label.setWordWrap(True)
        self.camera_optimal_label.setStyleSheet("color: #cccccc; font-family: monospace;")
        camera_optimal_layout.addWidget(self.camera_optimal_label)
        camera_optimal_group.setLayout(camera_optimal_layout)
        layout.addWidget(camera_optimal_group)
        
        # Log area
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
    
    def browse_directory(self):
        # First try to select a video file
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image Directory or Video File", 
            "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v *.webm);;All Files (*)"
        )
        
        if file_path:
            self.dir_edit.setText(file_path)
        else:
            # If canceled, try directory selection as fallback
            directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
            if directory:
                self.dir_edit.setText(directory)
    
    def browse_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON File", "", "JSON Files (*.json)"
        )
        if file_path:
            self.json_edit.setText(file_path)
    
    def _is_video_file(self, path):
        """Check if the given path is a video file"""
        if not path or not os.path.exists(path):
            return False
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}
        return Path(path).suffix.lower() in video_extensions
    
    def _show_downsample_dialog(self, video_path):
        """Show dialog to get downsample ratio from user"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Video Frame Extraction")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Message label
        video_name = Path(video_path).name
        output_dir = Path(video_path).parent / Path(video_path).stem
        message = QLabel(
            f"The input path is a video file:\n\n"
            f"{video_name}\n\n"
            f"The video will be decomposed to a directory of images with the same name as the video:\n\n"
            f"{output_dir}\n\n"
            f"Please specify the downsample ratio (save every Nth frame):"
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        
        # Downsample input
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Downsample ratio:"))
        downsample_spin = QSpinBox()
        downsample_spin.setMinimum(1)
        downsample_spin.setMaximum(1000)
        downsample_spin.setValue(DEFAULT_DOWNSAMPLE_RATIO)  # Default value
        input_layout.addWidget(downsample_spin)
        input_layout.addStretch()
        layout.addLayout(input_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.Accepted:
            return downsample_spin.value()
        return None
    
    def start_processing(self):
        # Validate inputs
        input_path = self.dir_edit.text().strip()
        if not input_path:
            QMessageBox.warning(self, "Error", "Please select a valid image directory or video file!")
            return
        
        image_dir = input_path
        
        # Check if input is a video file
        if self._is_video_file(input_path):
            # Check if extract_frames function is available
            if extract_frames is None:
                QMessageBox.critical(
                    self, 
                    "Error", 
                    "Video extraction function not available. Please ensure video2image.py is accessible."
                )
                return
            
            # Show dialog to get downsample ratio
            downsample = self._show_downsample_dialog(input_path)
            if downsample is None:
                # User canceled
                return
            
            try:
                # Extract frames from video
                self.log_message(f"Extracting frames from video: {input_path}")
                self.log_message(f"Downsample ratio: {downsample}")
                
                # Extract frames - output_dir will be automatically set based on video name
                saved_count = extract_frames(input_path, output_dir=None, downsample=downsample)
                
                # Update image_dir to point to the extracted directory
                video_path = Path(input_path)
                image_dir = str(video_path.parent / video_path.stem)
                
                self.log_message(f"Extracted {saved_count} frames to: {image_dir}")
                QMessageBox.information(
                    self,
                    "Video Extraction Complete",
                    f"Successfully extracted {saved_count} frames from the video.\n\n"
                    f"Images saved to: {image_dir}\n\n"
                    f"Continuing with processing..."
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Video Extraction Error",
                    f"Failed to extract frames from video:\n{str(e)}"
                )
                return
        
        # Now check if image_dir is a valid directory
        if not os.path.isdir(image_dir):
            QMessageBox.warning(self, "Error", f"Image directory does not exist: {image_dir}")
            return
        
        physical_length = self.physical_length_edit.text().strip()
        try:
            physical_length = float(physical_length)
            if physical_length <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid positive physical length!")
            return
        
        # Get ArUco size
        aruco_size_str = self.aruco_size_combo.currentText()
        aruco_size = int(aruco_size_str.split('x')[0])
        
        # Determine dictionary type based on size
        dict_type_map = {
            4: 'DICT_4X4_50',
            5: 'DICT_5X5_50',
            6: 'DICT_6X6_50'
        }
        dict_type = dict_type_map[aruco_size]
        
        # Parse camera intrinsic (required)
        fx = self.fx_edit.text().strip()
        fy = self.fy_edit.text().strip()
        cx = self.cx_edit.text().strip()
        cy = self.cy_edit.text().strip()
        
        if not (fx and fy and cx and cy):
            QMessageBox.warning(self, "Error", "Camera intrinsic parameters are required! Please enter fx, fy, cx, cy.")
            return
        
        try:
            fx = float(fx)
            fy = float(fy)
            cx = float(cx)
            cy = float(cy)
            if fx <= 0 or fy <= 0:
                raise ValueError("Focal lengths must be positive")
            camera_intrinsic = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
            self.log_message("Camera intrinsic provided. Bundle adjustment will use fixed intrinsic.")
        except ValueError as e:
            QMessageBox.warning(self, "Error", f"Invalid camera intrinsic values: {str(e)}")
            return
        
        # Get JSON path if provided
        json_path = self.json_edit.text().strip() if self.json_edit.text().strip() else None
        
        # Get Enable BA setting
        enable_ba = self.enable_ba_checkbox.isChecked()
        
        # Emit parameters
        params = {
            'image_dir': image_dir,
            'aruco_size': aruco_size,
            'physical_length': physical_length,
            'dict_type': dict_type,
            'camera_intrinsic': camera_intrinsic,
            'json_path': json_path,
            'enable_ba': enable_ba
        }
        
        self.parameters_ready.emit(params)
        self.start_btn.setEnabled(False)
    
    def log_message(self, message):
        self.log_text.append(message)
    
    def update_camera_intrinsic(self, camera_intrinsic):
        """Update camera intrinsic display after bundle adjustment"""
        if camera_intrinsic is not None:
            K = camera_intrinsic
            intrinsic_text = (
                f"fx: {K[0, 0]:.6f}\n"
                f"fy: {K[1, 1]:.6f}\n"
                f"cx: {K[0, 2]:.6f}\n"
                f"cy: {K[1, 2]:.6f}\n\n"
                f"Matrix:\n"
                f"[{K[0, 0]:.6f}   0   {K[0, 2]:.6f}]\n"
                f"[  0   {K[1, 1]:.6f}   {K[1, 2]:.6f}]\n"
                f"[  0     0     1   ]"
            )
            self.camera_optimal_label.setText(intrinsic_text)
        else:
            self.camera_optimal_label.setText("Not available yet")

