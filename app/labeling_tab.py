"""
Labeling tab for keypoint annotation
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QLineEdit, QPushButton, QTextEdit, QSlider,
                                QGroupBox, QMessageBox, QFileDialog, QComboBox,
                                QGridLayout, QSplitter, QSizePolicy, QSpinBox,
                                QDialog, QDialogButtonBox, QCheckBox)
from PySide6.QtCore import Qt, Signal, QSize, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QKeyEvent, QFont, QPolygon
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# Import default configuration and video extraction function
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from video2image import extract_frames
except ImportError:
    extract_frames = None
try:
    import default_cfg
    DEFAULT_FX = default_cfg.fx
    DEFAULT_FY = default_cfg.fy
    DEFAULT_CX = default_cfg.cx
    DEFAULT_CY = default_cfg.cy
    DEFAULT_PHYSICAL_LENGTH = default_cfg.physical_length
    DEFAULT_ARUCO_SIZE = default_cfg.aruco_size
    DEFAULT_DOWNSAMPLE_RATIO = default_cfg.downsample_ratio
    DEFAULT_ENABLE_BA = default_cfg.enable_ba
    DEFAULT_MAX_KEYPOINT_NUM = default_cfg.max_keypoint_num
    DEFAULT_MAX_CLS_ID = default_cfg.max_cls_id
    DEFAULT_VISIBILITY_THRESHOLD = default_cfg.visibility_threshold
except ImportError:
    # Fallback defaults if config file not found
    DEFAULT_FX = 1000.0
    DEFAULT_FY = 1000.0
    DEFAULT_CX = 540.0
    DEFAULT_CY = 360.0
    DEFAULT_PHYSICAL_LENGTH = 0.2
    DEFAULT_ARUCO_SIZE = '4x4'
    DEFAULT_DOWNSAMPLE_RATIO = 30  # Fallback default
    DEFAULT_ENABLE_BA = False  # Fallback default
    DEFAULT_MAX_KEYPOINT_NUM = 4  # Fallback default
    DEFAULT_MAX_CLS_ID = 3  # Fallback default
    DEFAULT_VISIBILITY_THRESHOLD = 85.0  # Fallback default

# Import utilities and constants
try:
    from .utils import triangulate_points, validate_frame_id, get_image_name_safe
    from .constants import (
        KEYPOINT_REMOVAL_THRESHOLD, KEYPOINT_DISPLAY_SIZE,
        KEYPOINT_LABEL_OFFSET_X, KEYPOINT_LABEL_OFFSET_Y,
        MARKER_AXIS_LENGTH_RATIO, MARKER_CORNER_SIZE, MARKER_BORDER_WIDTH
    )
    USE_UTILS = True
except ImportError:
    # Fallback if utils not available
    USE_UTILS = False
    KEYPOINT_REMOVAL_THRESHOLD = 10.0
    KEYPOINT_DISPLAY_SIZE = 5
    KEYPOINT_LABEL_OFFSET_X = 8
    KEYPOINT_LABEL_OFFSET_Y = -8
    MARKER_AXIS_LENGTH_RATIO = 0.3
    MARKER_CORNER_SIZE = 3
    MARKER_BORDER_WIDTH = 2
    
    def validate_frame_id(frame_id: int, num_frames: int) -> bool:
        return 0 <= frame_id < num_frames
    
    def get_image_name_safe(image_paths, frame_id: int) -> str:
        if not validate_frame_id(frame_id, len(image_paths)):
            raise IndexError(f"Frame ID {frame_id} out of range")
        return os.path.basename(image_paths[frame_id])


class ImageLabel(QWidget):
    """Custom widget for displaying image and handling clicks"""
    point_clicked = Signal(int, int)  # x, y
    keypoint_removed = Signal(int, int, int, int, int, bool)  # frame_id, object_id, keypoint_id, x, y, is_calculated
    
    def __init__(self):
        super().__init__()
        self.image = None
        self.pixmap = None
        self.keypoints = []  # List of (x, y, object_id, keypoint_id, frame_id)
        self.current_keypoint = None  # Currently selected keypoint
        self.current_frame_id = 0  # Current frame ID for removal
        # Initialize scale and offset for click handling
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
    
    def set_image(self, image):
        if image is not None:
            self.image = image.copy()
            h, w = image.shape[:2]
            # Convert BGR to RGB for Qt
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_image.data, w, h, w * 3, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(q_image)
        else:
            self.image = None
            self.pixmap = None
        self.update()
    
    def sizeHint(self):
        """Return preferred size"""
        if self.pixmap:
            return self.pixmap.size()
        return QSize(800, 600)
    
    def resizeEvent(self, event):
        """Handle widget resize"""
        super().resizeEvent(event)
        # Trigger repaint to update scaled image
        self.update()
    
    def add_keypoint(self, x, y, object_id, keypoint_id, frame_id, is_calculated=False, cls_id=1):
        """Add keypoint to display. is_calculated=True for triangulated keypoints"""
        self.keypoints.append((x, y, object_id, keypoint_id, frame_id, is_calculated, cls_id))
        self.update()
    
    def clear_keypoints(self):
        self.keypoints = []
        self.update()
    
    def set_keypoints(self, keypoints):
        self.keypoints = keypoints
        self.update()
    
    def set_current_keypoint(self, x, y):
        """Set the current keypoint being placed. Pass None, None to clear."""
        if x is None or y is None:
            self.current_keypoint = None
        else:
            self.current_keypoint = (x, y)
        self.update()
    
    def paintEvent(self, event):
        if self.pixmap is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Scale pixmap to fit widget
        scaled_pixmap = self.pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        x_offset = (self.width() - scaled_pixmap.width()) // 2
        y_offset = (self.height() - scaled_pixmap.height()) // 2
        
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
        
        # Store scale and offset for click handling
        self.scale_x = scaled_pixmap.width() / self.pixmap.width()
        self.scale_y = scaled_pixmap.height() / self.pixmap.height()
        self.offset_x = x_offset
        self.offset_y = y_offset
        
        # Group keypoints by object_id for current frame (for drawing connections)
        keypoints_by_object = {}  # {object_id: {keypoint_id: (wx, wy)}}
        cls_ids_by_object = {}  # {object_id: cls_id} - store cls_id for each object
        
        # First pass: collect keypoints and convert to widget coordinates
        for kp_data in self.keypoints:
            if len(kp_data) == 7:
                x, y, obj_id, kp_id, frame_id, is_calculated, cls_id = kp_data
            elif len(kp_data) == 6:
                x, y, obj_id, kp_id, frame_id, is_calculated = kp_data
                cls_id = 1  # Default cls_id for backward compatibility
            else:
                # Backward compatibility
                x, y, obj_id, kp_id, frame_id = kp_data
                is_calculated = False
                cls_id = 1  # Default cls_id
            
            # Only process keypoints in current frame
            if frame_id == self.current_frame_id:
                # Convert to widget coordinates
                wx = int(x * self.scale_x + self.offset_x)
                wy = int(y * self.scale_y + self.offset_y)
                
                # Group by object_id
                if obj_id not in keypoints_by_object:
                    keypoints_by_object[obj_id] = {}
                keypoints_by_object[obj_id][kp_id] = (wx, wy)
                # Store cls_id for this object (use the first one we encounter)
                if obj_id not in cls_ids_by_object:
                    cls_ids_by_object[obj_id] = cls_id
        
        # Draw filled polygon for objects with more than 2 keypoints (before drawing lines)
        very_light_red = QColor(255, 200, 200, 50)  # Very light red with transparency
        
        for obj_id, kp_dict in keypoints_by_object.items():
            num_keypoints = len(kp_dict)
            if num_keypoints > 2:
                # Sort keypoints by keypoint_id to get ordered polygon points
                sorted_kp_ids = sorted(kp_dict.keys())
                polygon_points = []
                for kp_id in sorted_kp_ids:
                    wx, wy = kp_dict[kp_id]
                    polygon_points.append(QPoint(wx, wy))
                
                # Draw filled polygon
                polygon = QPolygon(polygon_points)
                painter.setPen(QPen(QColor(255, 200, 200, 100), 1))  # Light red border
                painter.setBrush(very_light_red)
                painter.drawPolygon(polygon)
        
        # Draw lines connecting keypoints for each object (before drawing keypoints)
        # Use light green color
        light_green = QColor(144, 238, 144)  # Light green
        painter.setPen(QPen(light_green, 2))
        
        for obj_id, kp_dict in keypoints_by_object.items():
            # Sort keypoints by keypoint_id
            sorted_kp_ids = sorted(kp_dict.keys())
            
            # Draw lines connecting consecutive keypoints
            for i in range(len(sorted_kp_ids) - 1):
                kp_id1 = sorted_kp_ids[i]
                kp_id2 = sorted_kp_ids[i + 1]
                point1 = kp_dict[kp_id1]
                point2 = kp_dict[kp_id2]
                painter.drawLine(point1[0], point1[1], point2[0], point2[1])
            
            # If we have keypoint with id max_keypoint_num and id 0 (or id 1 if 0 doesn't exist), connect them (close the loop)
            # Check for id 0 first, then fall back to id 1
            first_kp_id = 0 if 0 in kp_dict else (1 if 1 in kp_dict else None)
            if DEFAULT_MAX_KEYPOINT_NUM in kp_dict and first_kp_id is not None:
                point_max = kp_dict[DEFAULT_MAX_KEYPOINT_NUM]
                point_first = kp_dict[first_kp_id]
                painter.drawLine(point_max[0], point_max[1], point_first[0], point_first[1])
        
        # Draw class ID at average keypoint location for each object
        for obj_id, kp_dict in keypoints_by_object.items():
            # Calculate average position of keypoints
            if kp_dict:
                avg_x = sum(wx for wx, wy in kp_dict.values()) / len(kp_dict)
                avg_y = sum(wy for wx, wy in kp_dict.values()) / len(kp_dict)
                
                # Get cls_id for this object
                cls_id = cls_ids_by_object.get(obj_id, 1)
                
                # Draw class ID text
                painter.setFont(QFont("Arial", 12, QFont.Bold))
                text = str(cls_id)
                # Calculate text bounding rect for centering
                text_rect = painter.fontMetrics().boundingRect(text)
                # Create background rectangle with padding
                bg_rect = text_rect.adjusted(-3, -2, 3, 2)
                # Center the rectangle at average position
                bg_x = int(avg_x) - bg_rect.width() // 2
                bg_y = int(avg_y) - bg_rect.height() // 2
                bg_rect.moveTo(bg_x, bg_y)
                # Draw semi-transparent black background
                painter.fillRect(bg_rect, QColor(0, 0, 0, 180))
                # Draw text in yellow, centered
                painter.setPen(QPen(QColor(255, 255, 0), 1))  # Yellow text
                text_x = int(avg_x) - text_rect.width() // 2
                text_y = int(avg_y) + text_rect.height() // 2 - painter.fontMetrics().descent()
                painter.drawText(text_x, text_y, text)
        
        # Draw keypoints
        for kp_data in self.keypoints:
            if len(kp_data) == 7:
                x, y, obj_id, kp_id, frame_id, is_calculated, cls_id = kp_data
            elif len(kp_data) == 6:
                x, y, obj_id, kp_id, frame_id, is_calculated = kp_data
            else:
                # Backward compatibility
                x, y, obj_id, kp_id, frame_id = kp_data
                is_calculated = False
            
            # Convert to widget coordinates
            wx = int(x * self.scale_x + self.offset_x)
            wy = int(y * self.scale_y + self.offset_y)
            
            # Different colors for calculated vs user-labeled
            if is_calculated:
                # Calculated keypoints: blue
                painter.setPen(QPen(QColor(0, 150, 255), 2))
                painter.setBrush(QColor(0, 150, 255, 80))
            else:
                # User-labeled keypoints: green
                painter.setPen(QPen(QColor(0, 255, 0), 3))
                painter.setBrush(QColor(0, 255, 0, 100))
            
            size = KEYPOINT_DISPLAY_SIZE
            painter.drawEllipse(wx - size, wy - size, size * 2, size * 2)
            
            # Draw label
            label = f"O{obj_id}_K{kp_id}"
            if is_calculated:
                label += "*"  # Mark calculated keypoints
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(wx + KEYPOINT_LABEL_OFFSET_X, wy + KEYPOINT_LABEL_OFFSET_Y, label)
        
        # Draw current keypoint (if being placed)
        if self.current_keypoint:
            x, y = self.current_keypoint
            wx = int(x * self.scale_x + self.offset_x)
            wy = int(y * self.scale_y + self.offset_y)
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.setBrush(QColor(255, 0, 0, 100))
            painter.drawEllipse(wx - 5, wy - 5, 10, 10)
        
        # Draw ArUco markers: corners and RGB axes
        if self.camera_intrinsic is not None:
            for marker_info in self.aruco_markers:
                marker_id = marker_info['marker_id']
                corners = marker_info['corners']  # 4x2 array
                rvec = marker_info['rvec']
                tvec = marker_info['tvec']
                
                # Draw marker corners (connect them to form a square)
                corner_points = []
                for corner in corners:
                    wx = int(corner[0] * self.scale_x + self.offset_x)
                    wy = int(corner[1] * self.scale_y + self.offset_y)
                    corner_points.append(QPoint(wx, wy))
                
                # Draw marker border
                painter.setPen(QPen(QColor(255, 255, 0), 2))  # Yellow border
                for i in range(4):
                    painter.drawLine(corner_points[i], corner_points[(i+1)%4])
                
                # Draw corner points
                painter.setPen(QPen(QColor(255, 255, 0), MARKER_BORDER_WIDTH))
                painter.setBrush(QColor(255, 255, 0))
                for point in corner_points:
                    painter.drawEllipse(point, MARKER_CORNER_SIZE, MARKER_CORNER_SIZE)
                
                # Draw RGB coordinate axes
                # Define axis endpoints in marker frame (3D)
                axis_length = self.marker_size * MARKER_AXIS_LENGTH_RATIO
                axis_points_3d = np.array([
                    [0, 0, 0],  # Origin
                    [axis_length, 0, 0],  # X axis (red)
                    [0, axis_length, 0],  # Y axis (green)
                    [0, 0, -axis_length]  # Z axis (blue) - negative because camera looks down -Z
                ], dtype=np.float32)
                
                # Project axes to 2D
                projected_axes, _ = cv2.projectPoints(
                    axis_points_3d,
                    rvec,
                    tvec,
                    self.camera_intrinsic,
                    None
                )
                projected_axes = projected_axes.reshape(-1, 2)
                
                origin_2d = projected_axes[0]
                x_axis_2d = projected_axes[1]
                y_axis_2d = projected_axes[2]
                z_axis_2d = projected_axes[3]
                
                # Convert to widget coordinates
                def to_widget(pt):
                    return QPoint(int(pt[0] * self.scale_x + self.offset_x),
                                int(pt[1] * self.scale_y + self.offset_y))
                
                origin_w = to_widget(origin_2d)
                x_w = to_widget(x_axis_2d)
                y_w = to_widget(y_axis_2d)
                z_w = to_widget(z_axis_2d)
                
                # Draw X axis (red)
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.drawLine(origin_w, x_w)
                painter.drawEllipse(x_w, 2, 2)
                
                # Draw Y axis (green)
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.drawLine(origin_w, y_w)
                painter.drawEllipse(y_w, 2, 2)
                
                # Draw Z axis (blue)
                painter.setPen(QPen(QColor(0, 0, 255), 2))
                painter.drawLine(origin_w, z_w)
                painter.drawEllipse(z_w, 2, 2)
                
                # Draw marker ID label
                label_pos = corner_points[0]  # Top-left corner
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.setFont(QFont("Arial", 10, QFont.Bold))
                painter.drawText(label_pos.x() + 5, label_pos.y() - 5, f"ArUco {marker_id}")
    
    def set_current_frame(self, frame_id):
        """Set current frame ID for keypoint removal"""
        self.current_frame_id = frame_id
    
    def set_aruco_markers(self, markers, camera_intrinsic, marker_size):
        """Set ArUco markers to display for current frame"""
        self.aruco_markers = markers
        self.camera_intrinsic = camera_intrinsic
        self.marker_size = marker_size
        self.update()
    
    def mousePressEvent(self, event):
        if self.pixmap is None:
            return
        
        # Convert widget coordinates to image coordinates
        x = (event.x() - self.offset_x) / self.scale_x
        y = (event.y() - self.offset_y) / self.scale_y
        
        # Check bounds
        if 0 <= x < self.pixmap.width() and 0 <= y < self.pixmap.height():
            if event.button() == Qt.RightButton:
                # Right-click: find and remove closest keypoint
                self.remove_closest_keypoint(x, y)
            else:
                # Left-click: emit point clicked
                self.point_clicked.emit(int(x), int(y))
    
    def remove_closest_keypoint(self, x, y):
        """Remove closest keypoint within threshold (both user-labeled and calculated)"""
        min_dist = KEYPOINT_REMOVAL_THRESHOLD
        closest_kp = None
        
        for kp in self.keypoints:
            if len(kp) == 7:
                kp_x, kp_y, obj_id, kp_id, frame_id, is_calculated, cls_id = kp
            elif len(kp) == 6:
                kp_x, kp_y, obj_id, kp_id, frame_id, is_calculated = kp
            else:
                kp_x, kp_y, obj_id, kp_id, frame_id = kp
                is_calculated = False
            
            # Check all keypoints in current frame (both user-labeled and calculated)
            if frame_id == self.current_frame_id:
                dist = np.sqrt((x - kp_x)**2 + (y - kp_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_kp = kp
        
        if closest_kp:
            if len(closest_kp) == 7:
                kp_x, kp_y, obj_id, kp_id, frame_id, is_calculated, cls_id = closest_kp
            elif len(closest_kp) == 6:
                kp_x, kp_y, obj_id, kp_id, frame_id, is_calculated = closest_kp
            else:
                kp_x, kp_y, obj_id, kp_id, frame_id = closest_kp
                is_calculated = False
            # Remove from display
            self.keypoints.remove(closest_kp)
            self.update()
            # Emit signal for removal (include is_calculated flag)
            self.keypoint_removed.emit(frame_id, obj_id, kp_id, int(kp_x), int(kp_y), is_calculated)


class LabelingTab(QWidget):
    keypoint_labeled = Signal(int, int, int, list, int)  # frame_id, object_id, keypoint_id, [x, y], cls_id
    keypoint_removed = Signal(int, int, int, int, int, bool)  # frame_id, object_id, keypoint_id, x, y, is_calculated
    done_clicked = Signal()
    save_label_clicked = Signal()  # Signal for save label
    save_yolo_clicked = Signal()  # Signal for save YOLO format
    parameters_ready = Signal(dict)  # Signal for parameters
    exit_requested = Signal()  # Signal for exit request
    
    def __init__(self):
        super().__init__()
        self.images = []
        self.image_paths = []
        self.aruco_poses = {}
        self.camera_poses = {}  # Per-frame camera poses
        self.camera_intrinsic = None
        self.frame_detections = {}  # {frame_id: [detection1, ...]} - ArUco detections per frame
        self.marker_size = 0.05  # Physical size of ArUco marker
        self.labeling_data = {}
        self.calculated_2d = {}  # {frame_id: {object_id: {keypoint_id: [x, y]}}} - calculated from triangulation
        self.calculated_3d = {}  # {(object_id, keypoint_id): [x, y, z]} - triangulated 3D locations
        self.calculated_visibility = {}  # {(frame_id, object_id, keypoint_id): bool} - visibility for triangulated keypoints loaded from JSON
        self.current_frame = 0
        self.is_setup = False  # Track if labeling is ready
        self.saved_json_path = None  # Track if JSON has been saved
        self.saved_yolo_path = None  # Track if YOLO format has been saved
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus
        self.init_ui()
    
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Use splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left side: Keypoint input widget
        left_input_widget = QWidget()
        left_input_layout = QVBoxLayout(left_input_widget)
        left_input_layout.setContentsMargins(0, 0, 0, 0)
        
        # Object and keypoint ID input
        id_group = QGroupBox("Keypoint Information")
        id_layout = QVBoxLayout()
        
        id_layout.addWidget(QLabel("Object ID:"))
        self.object_id_spin = QSpinBox()
        self.object_id_spin.setMinimum(1)
        self.object_id_spin.setMaximum(999)
        self.object_id_spin.setValue(1)
        id_layout.addWidget(self.object_id_spin)
        
        id_layout.addWidget(QLabel("Keypoint ID:"))
        self.keypoint_id_spin = QSpinBox()
        self.keypoint_id_spin.setMinimum(1)
        self.keypoint_id_spin.setMaximum(DEFAULT_MAX_KEYPOINT_NUM)
        self.keypoint_id_spin.setValue(1)
        id_layout.addWidget(self.keypoint_id_spin)
        
        id_layout.addWidget(QLabel("Class ID:"))
        self.cls_id_spin = QSpinBox()
        self.cls_id_spin.setMinimum(1)
        self.cls_id_spin.setMaximum(DEFAULT_MAX_CLS_ID)
        self.cls_id_spin.setValue(1)
        id_layout.addWidget(self.cls_id_spin)
        
        # Recommendation label
        self.recommendation_label = QLabel("")
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setStyleSheet("color: #4ec9b0;")
        id_layout.addWidget(self.recommendation_label)
        
        # Show triangulated keypoints checkbox
        self.show_triangulated_checkbox = QCheckBox("Show Triangulated Keypoints")
        self.show_triangulated_checkbox.setChecked(True)  # Default to showing them
        self.show_triangulated_checkbox.stateChanged.connect(self.on_triangulated_visibility_changed)
        id_layout.addWidget(self.show_triangulated_checkbox)
        
        # Check visibility checkbox
        self.check_visibility_checkbox = QCheckBox("Check Visibility")
        self.check_visibility_checkbox.setChecked(True)  # Default to enabled
        self.check_visibility_checkbox.stateChanged.connect(self.on_visibility_check_changed)
        id_layout.addWidget(self.check_visibility_checkbox)
        
        # Save Label button
        self.save_label_btn = QPushButton("Save Label (json)")
        self.save_label_btn.clicked.connect(self.on_save_label_clicked)
        self.save_label_btn.setEnabled(False)  # Disabled until labeling is ready
        id_layout.addWidget(self.save_label_btn)
        
        # Save YOLO pose format button
        self.save_yolo_btn = QPushButton("Save Label (YOLO pose)")
        self.save_yolo_btn.clicked.connect(self.on_save_yolo_clicked)
        self.save_yolo_btn.setEnabled(False)  # Disabled until labeling is ready
        id_layout.addWidget(self.save_yolo_btn)
        
        id_group.setLayout(id_layout)
        left_input_layout.addWidget(id_group)
        
        left_input_layout.addStretch()
        
        # Exit button at the bottom
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.on_exit_clicked)
        self.exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
            QPushButton:pressed {
                background-color: #8b0000;
            }
        """)
        left_input_layout.addWidget(self.exit_btn)
        
        splitter.addWidget(left_input_widget)
        
        # Middle: Image display
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        # Image label - make it expandable
        self.image_label = ImageLabel()
        self.image_label.point_clicked.connect(self.on_image_clicked)
        self.image_label.keypoint_removed.connect(self.on_keypoint_removed)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_layout.addWidget(self.image_label, 1)  # Stretch factor 1
        
        # Timeline slider
        timeline_group = QGroupBox("Timeline")
        timeline_layout = QVBoxLayout()
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.valueChanged.connect(self.on_frame_changed)
        timeline_layout.addWidget(self.timeline_slider)
        
        self.frame_label = QLabel("Frame: 0 / 0")
        timeline_layout.addWidget(self.frame_label)
        timeline_group.setLayout(timeline_layout)
        image_layout.addWidget(timeline_group, 0)
        
        splitter.addWidget(image_widget)
        
        # Right side: Split into parameter panel and info panel
        right_splitter = QSplitter(Qt.Vertical)
        
        # Upper right: Parameter panel
        self.parameter_widget = self.create_parameter_widget()
        right_splitter.addWidget(self.parameter_widget)
        
        # Lower right: Info panel
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        # Log area
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        info_layout.addWidget(log_group, 1)  # Give log area stretch
        
        right_splitter.addWidget(info_widget)
        
        # Set splitter proportions (40% parameters, 60% info)
        right_splitter.setSizes([400, 600])
        
        splitter.addWidget(right_splitter)
        
        # Set splitter proportions (15% left input, 60% image, 25% right panel)
        splitter.setSizes([200, 1000, 400])
    
    def create_parameter_widget(self):
        """Create parameter input widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Image directory selection
        dir_group = QGroupBox("Image Directory / Video File")
        dir_layout = QVBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Select directory containing images or video file...")
        dir_layout.addWidget(self.dir_edit)
        
        # Two separate buttons for video and directory
        button_layout = QHBoxLayout()
        video_browse_btn = QPushButton("Browse Video...")
        video_browse_btn.clicked.connect(self.browse_video)
        dir_browse_btn = QPushButton("Browse Directory...")
        dir_browse_btn.clicked.connect(self.browse_directory)
        button_layout.addWidget(video_browse_btn)
        button_layout.addWidget(dir_browse_btn)
        dir_layout.addLayout(button_layout)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # ArUco parameters
        aruco_group = QGroupBox("ArUco Marker Parameters (Optional)")
        aruco_layout = QGridLayout()
        
        # ArUco size selection
        aruco_layout.addWidget(QLabel("ArUco Size:"), 0, 0)
        self.aruco_size_combo = QComboBox()
        self.aruco_size_combo.addItems(["4x4", "5x5", "6x6"])
        # Set default value from config
        default_size_idx = self.aruco_size_combo.findText(DEFAULT_ARUCO_SIZE)
        if default_size_idx >= 0:
            self.aruco_size_combo.setCurrentIndex(default_size_idx)
        aruco_layout.addWidget(self.aruco_size_combo, 0, 1)
        
        # Physical length (optional, default from config)
        aruco_layout.addWidget(QLabel("Physical Length (meters):"), 1, 0)
        self.physical_length_edit = QLineEdit()
        self.physical_length_edit.setPlaceholderText(f"Default: {DEFAULT_PHYSICAL_LENGTH}")
        self.physical_length_edit.setText(str(DEFAULT_PHYSICAL_LENGTH))  # Set default value from config
        aruco_layout.addWidget(self.physical_length_edit, 1, 1)
        
        aruco_group.setLayout(aruco_layout)
        layout.addWidget(aruco_group)
        
        # Camera intrinsic (optional)
        camera_group = QGroupBox("Camera Intrinsic (Optional)")
        camera_layout = QGridLayout()
        
        camera_layout.addWidget(QLabel("fx:"), 0, 0)
        self.fx_edit = QLineEdit()
        self.fx_edit.setPlaceholderText(f"Default: {DEFAULT_FX}")
        self.fx_edit.setText(str(DEFAULT_FX))  # Set default value from config
        camera_layout.addWidget(self.fx_edit, 0, 1)
        
        camera_layout.addWidget(QLabel("fy:"), 0, 2)
        self.fy_edit = QLineEdit()
        self.fy_edit.setPlaceholderText(f"Default: {DEFAULT_FY}")
        self.fy_edit.setText(str(DEFAULT_FY))  # Set default value from config
        camera_layout.addWidget(self.fy_edit, 0, 3)
        
        camera_layout.addWidget(QLabel("cx:"), 1, 0)
        self.cx_edit = QLineEdit()
        self.cx_edit.setPlaceholderText(f"Default: {DEFAULT_CX}")
        self.cx_edit.setText(str(DEFAULT_CX))  # Set default value from config
        camera_layout.addWidget(self.cx_edit, 1, 1)
        
        camera_layout.addWidget(QLabel("cy:"), 1, 2)
        self.cy_edit = QLineEdit()
        self.cy_edit.setPlaceholderText(f"Default: {DEFAULT_CY}")
        self.cy_edit.setText(str(DEFAULT_CY))  # Set default value from config
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
        
        layout.addStretch()
        
        return widget
    
    def browse_video(self):
        """Browse for video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Video File", 
            "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v *.webm);;All Files (*)"
        )
        
        if file_path:
            self.dir_edit.setText(file_path)
    
    def browse_directory(self):
        """Browse for image directory"""
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
        # Clear previous session data when starting a new session
        if self.is_setup:
            # Check if there's any labeling data or calculated data
            has_data = (self.labeling_data or self.calculated_2d or self.calculated_3d)
            
            # Check if data has been saved (either JSON or YOLO format)
            json_saved = self.saved_json_path is not None
            yolo_saved = self.saved_yolo_path is not None
            data_saved = json_saved or yolo_saved
            
            # If there's data but nothing saved, warn user
            if has_data and not data_saved:
                reply = QMessageBox.warning(
                    self,
                    "Unsaved Changes",
                    "You have unsaved labeling data. Please save your work before starting a new session.\n\n"
                    "Click 'Yes' to continue without saving (this will clear all current data), "
                    "or 'No' to cancel and save your work.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    return  # User cancelled
            elif has_data:
                # Data exists but has been saved - just confirm
                reply = QMessageBox.question(
                    self,
                    "Start New Session",
                    "Starting a new labeling session will clear all current labeling data.\n\n"
                    "Are you sure you want to continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    return  # User cancelled
            
            # Clear previous session data
            self.reset_session()
        
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
        
        # Get physical length (use default from config if empty)
        physical_length_str = self.physical_length_edit.text().strip()
        if physical_length_str:
            try:
                physical_length = float(physical_length_str)
                if physical_length <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Error", "Please enter a valid positive physical length!")
                return
        else:
            physical_length = DEFAULT_PHYSICAL_LENGTH  # Default value from config
        
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
        
        # Parse camera intrinsic (use defaults if empty)
        fx_str = self.fx_edit.text().strip()
        fy_str = self.fy_edit.text().strip()
        cx_str = self.cx_edit.text().strip()
        cy_str = self.cy_edit.text().strip()
        
        # Use defaults from config if empty
        try:
            fx = float(fx_str) if fx_str else DEFAULT_FX
            fy = float(fy_str) if fy_str else DEFAULT_FY
            cx = float(cx_str) if cx_str else DEFAULT_CX
            cy = float(cy_str) if cy_str else DEFAULT_CY
            
            if fx <= 0 or fy <= 0:
                raise ValueError("Focal lengths must be positive")
            camera_intrinsic = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float64)
            if fx_str or fy_str or cx_str or cy_str:
                self.log_message("Using provided camera intrinsic values.")
            else:
                self.log_message(f"Using default camera intrinsic values (fx={DEFAULT_FX}, fy={DEFAULT_FY}, cx={DEFAULT_CX}, cy={DEFAULT_CY}).")
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
        self.log_message("Starting new processing session...")
    
    def setup(self, images, image_paths, aruco_poses, camera_poses, camera_intrinsic, labeling_data, frame_aruco_count=None, frame_detections=None, marker_size=0.05):
        """Initialize with data from main window"""
        self.images = images
        self.image_paths = image_paths
        self.aruco_poses = aruco_poses
        self.camera_poses = camera_poses  # Per-frame camera poses
        self.camera_intrinsic = camera_intrinsic
        self.frame_detections = frame_detections or {}
        self.marker_size = marker_size
        self.labeling_data = labeling_data
        self.frame_aruco_count = frame_aruco_count or {}
        self.is_setup = True
        
        # Enable labeling controls
        self.save_label_btn.setEnabled(True)
        self.save_yolo_btn.setEnabled(True)
        
        # Setup timeline
        self.timeline_slider.setMaximum(len(images) - 1)
        self.current_frame = 0
        self.update_frame_display()
        
        self.log_message(f"Ready for labeling. {len(images)} frames loaded.")
        
        # Update IDs for current frame
        self.update_ids_for_current_frame()
        
        # Check all existing keypoints for triangulation
        self.check_all_keypoints_for_triangulation()
    
    def keyPressEvent(self, event):
        """Handle keyboard navigation"""
        if event.key() == Qt.Key_Left:
            if self.current_frame > 0:
                self.current_frame -= 1
                self.timeline_slider.setValue(self.current_frame)
                # on_frame_changed will be called automatically, which updates IDs
        elif event.key() == Qt.Key_Right:
            if self.current_frame < len(self.images) - 1:
                self.current_frame += 1
                self.timeline_slider.setValue(self.current_frame)
                # on_frame_changed will be called automatically, which updates IDs
        else:
            super().keyPressEvent(event)
    
    def on_frame_changed(self, value):
        """Handle timeline slider change"""
        old_frame = self.current_frame
        self.current_frame = value
        self.image_label.set_current_frame(value)
        print(f"[FRAME] Changed from frame {old_frame} to frame {value}")
        
        # Update object_id and keypoint_id to max values in current frame
        self.update_ids_for_current_frame()
        
        self.update_frame_display()
    
    def update_ids_for_current_frame(self):
        """Update object_id and keypoint_id to max values of manually labeled keypoints in current frame"""
        frame_id = self.current_frame
        max_object_id = 0
        max_keypoint_id = 0
        
        # Find max object_id and keypoint_id in current frame (only manual labels)
        if frame_id in self.labeling_data:
            for obj_id, keypoints in self.labeling_data[frame_id].items():
                if obj_id > max_object_id:
                    max_object_id = obj_id
                for kp_id, data in keypoints.items():
                    if data.get('2d') is not None:  # Only count manually labeled keypoints
                        if kp_id > max_keypoint_id:
                            max_keypoint_id = kp_id
        
        # Set IDs: if no manual labels, use (1, 1), otherwise use max values
        if max_object_id == 0 and max_keypoint_id == 0:
            self.object_id_spin.setValue(1)
            self.keypoint_id_spin.setValue(1)
        else:
            self.object_id_spin.setValue(max_object_id)
            self.keypoint_id_spin.setValue(max_keypoint_id)
        
        print(f"[FRAME] Updated IDs: Object={self.object_id_spin.value()}, Keypoint={self.keypoint_id_spin.value()}")
    
    def on_keypoint_removed(self, frame_id, object_id, keypoint_id, x, y, is_calculated):
        """Handle keypoint removal"""
        print(f"[REMOVAL] Removing keypoint: Frame {frame_id}, Object {object_id}, Keypoint {keypoint_id} at ({x}, {y}), is_calculated={is_calculated}")
        
        removed_cls_id = 1  # Default cls_id
        if is_calculated:
            # Remove from calculated_2d
            if frame_id in self.calculated_2d:
                if object_id in self.calculated_2d[frame_id]:
                    if keypoint_id in self.calculated_2d[frame_id][object_id]:
                        del self.calculated_2d[frame_id][object_id][keypoint_id]
                        # Clean up empty dictionaries
                        if not self.calculated_2d[frame_id][object_id]:
                            del self.calculated_2d[frame_id][object_id]
                        if not self.calculated_2d[frame_id]:
                            del self.calculated_2d[frame_id]
                        
                        # Also remove from calculated_3d if exists
                        if (object_id, keypoint_id) in self.calculated_3d:
                            del self.calculated_3d[(object_id, keypoint_id)]
                        
                        print(f"[REMOVAL] Calculated keypoint removed successfully")
                        self.log_message(f"Removed calculated keypoint: Frame {frame_id}, Object {object_id}, Keypoint {keypoint_id}")
        else:
            # Remove from user-labeled data
            if frame_id in self.labeling_data:
                if object_id in self.labeling_data[frame_id]:
                    if keypoint_id in self.labeling_data[frame_id][object_id]:
                        # Get cls_id before deletion
                        removed_cls_id = self.labeling_data[frame_id][object_id][keypoint_id].get('cls_id', 1)
                        del self.labeling_data[frame_id][object_id][keypoint_id]
                        # Clean up empty dictionaries
                        if not self.labeling_data[frame_id][object_id]:
                            del self.labeling_data[frame_id][object_id]
                        if not self.labeling_data[frame_id]:
                            del self.labeling_data[frame_id]
                        
                        print(f"[REMOVAL] User-labeled keypoint removed successfully")
                        self.log_message(f"Removed keypoint: Frame {frame_id}, Object {object_id}, Keypoint {keypoint_id}")
                        
                        # Re-check triangulation after removal
                        print(f"[REMOVAL] Re-checking triangulation after removal...")
                        self.check_and_triangulate_keypoint(object_id, keypoint_id)
        
        # Set object_id, keypoint_id, and cls_id to the values of the removed keypoint
        self.object_id_spin.setValue(object_id)
        self.keypoint_id_spin.setValue(keypoint_id)
        if not is_calculated:
            self.cls_id_spin.setValue(removed_cls_id)
        
        print(f"[REMOVAL] Updated IDs to removed keypoint values: Object={object_id}, Keypoint={keypoint_id}, Class={removed_cls_id if not is_calculated else 'N/A'}")
        
        # Update display
        self.update_frame_display()
        
        # Emit signal to main window
        self.keypoint_removed.emit(frame_id, object_id, keypoint_id, x, y, is_calculated)
    
    def on_save_label_clicked(self):
        """Handle Save Label button click"""
        self.save_label_clicked.emit()
    
    def on_save_yolo_clicked(self):
        """Handle Save YOLO pose format button click"""
        self.save_yolo_clicked.emit()
    
    def on_exit_clicked(self):
        """Handle Exit button click"""
        self.exit_requested.emit()
    
    def set_saved_json_path(self, json_path):
        """Set the saved JSON path to track if data has been saved"""
        self.saved_json_path = json_path
    
    def set_saved_yolo_path(self, yolo_path):
        """Set the saved YOLO path to track if data has been saved"""
        self.saved_yolo_path = yolo_path
    
    def update_frame_display(self):
        """Update image and keypoints display"""
        if not self.images:
            return
        
        # Validate frame ID
        if not validate_frame_id(self.current_frame, len(self.images)):
            self.log_message(f"Warning: Invalid frame ID {self.current_frame}, resetting to 0")
            self.current_frame = 0
            self.timeline_slider.setValue(0)
            return
        
        frame_id = self.current_frame
        print(f"[DISPLAY] Updating frame {frame_id} display...")
        
        # Display image
        self.image_label.set_image(self.images[self.current_frame])
        
        # Get ArUco markers for current frame
        aruco_markers = []
        if frame_id in self.frame_detections:
            for det in self.frame_detections[frame_id]:
                aruco_markers.append({
                    'marker_id': det['marker_id'],
                    'corners': det['corners'],
                    'rvec': np.array(det['rvec']),
                    'tvec': np.array(det['tvec'])
                })
        
        # Set ArUco markers for display
        self.image_label.set_aruco_markers(aruco_markers, self.camera_intrinsic, self.marker_size)
        
        # Display keypoints for current frame
        keypoints_to_show = []
        user_labeled_count = 0
        calc_count = 0
        
        # Track which objects are visible (to avoid checking multiple times)
        visible_objects = {}  # {object_id: bool}
        
        # Add user-labeled keypoints
        if frame_id in self.labeling_data:
            for obj_id, keypoints in self.labeling_data[frame_id].items():
                # Check visibility for this object
                if obj_id not in visible_objects:
                    visible_objects[obj_id] = self.check_object_visibility(frame_id, obj_id)
                
                if visible_objects[obj_id]:
                    for kp_id, data in keypoints.items():
                        if data.get('2d') is not None:
                            x, y = data['2d']
                            cls_id = data.get('cls_id', 1)  # Get cls_id, default to 1
                            keypoints_to_show.append((x, y, obj_id, kp_id, frame_id, False, cls_id))
                            user_labeled_count += 1
                            print(f"[DISPLAY]   User-labeled: Object {obj_id}, Keypoint {kp_id} at ({x:.2f}, {y:.2f})")
        
        # Add calculated keypoints (from triangulation) - only if checkbox is checked
        if self.show_triangulated_checkbox.isChecked() and frame_id in self.calculated_2d:
            for obj_id, keypoints in self.calculated_2d[frame_id].items():
                # Check visibility for this object
                if obj_id not in visible_objects:
                    visible_objects[obj_id] = self.check_object_visibility(frame_id, obj_id)
                
                if visible_objects[obj_id]:
                    for kp_id, point_2d in keypoints.items():
                        # Only show if not already labeled by user
                        is_user_labeled = (frame_id in self.labeling_data and 
                                          obj_id in self.labeling_data[frame_id] and
                                          kp_id in self.labeling_data[frame_id][obj_id])
                        
                        if not is_user_labeled:
                            x, y = point_2d
                            # Get cls_id from any manually labeled keypoint of this object across all frames
                            cls_id = self.get_object_cls_id(obj_id)
                            keypoints_to_show.append((x, y, obj_id, kp_id, frame_id, True, cls_id))
                            calc_count += 1
                            print(f"[DISPLAY]   Calculated: Object {obj_id}, Keypoint {kp_id} at ({x:.2f}, {y:.2f})")
        
        print(f"[DISPLAY]   Total keypoints: {user_labeled_count} user-labeled, {calc_count} calculated")
        print(f"[DISPLAY]   ArUco markers: {len(aruco_markers)}")
        
        if calc_count > 0:
            self.log_message(f"Frame {frame_id}: Showing {calc_count} calculated keypoint(s)")
        
        self.image_label.set_keypoints(keypoints_to_show)
        
        # Update frame label
        self.frame_label.setText(f"Frame: {self.current_frame + 1} / {len(self.images)}")
    
    def on_triangulated_visibility_changed(self, state):
        """Handle triangulated keypoints visibility checkbox change"""
        # Update display when checkbox state changes
        self.update_frame_display()
    
    def on_visibility_check_changed(self, state):
        """Handle visibility check checkbox change"""
        # Update display when checkbox state changes
        self.update_frame_display()
    
    def get_object_cls_id(self, object_id):
        """
        Get the class ID for an object by searching all frames for manually labeled keypoints.
        
        Args:
            object_id: Object ID to get class ID for
            
        Returns:
            int: Class ID for the object, or 1 if not found
        """
        # Search all frames for this object
        for frame_id, objects in self.labeling_data.items():
            if object_id in objects:
                # Get cls_id from any keypoint of this object in this frame
                for kp_id, data in objects[object_id].items():
                    # Return the first cls_id we find (defaults to 1 if not set)
                    return data.get('cls_id', 1)
        # Default to 1 if object not found in any frame
        return 1
    
    def check_object_visibility(self, frame_id, object_id):
        """
        Check if an object is visible in the given frame.
        
        Args:
            frame_id: Frame ID to check
            object_id: Object ID to check
            
        Returns:
            bool: True if object is visible, False if invisible
        """
        # Skip if visibility check is disabled
        if not self.check_visibility_checkbox.isChecked():
            return True
        
        # Collect all keypoints for this object in this frame
        keypoints_3d = {}  # {keypoint_id: [x, y, z]}
        
        # Get user-labeled keypoints with 3D positions
        if frame_id in self.labeling_data:
            if object_id in self.labeling_data[frame_id]:
                # Get camera pose for calculating 3D if needed
                camera_pose = None
                if frame_id in self.camera_poses:
                    camera_pose = self.camera_poses[frame_id]
                
                for kp_id, data in self.labeling_data[frame_id][object_id].items():
                    if data.get('3d') is not None:
                        keypoints_3d[kp_id] = np.array(data['3d'])
                    # Also check if we have calculated 3D for this keypoint
                    elif (object_id, kp_id) in self.calculated_3d:
                        keypoints_3d[kp_id] = np.array(self.calculated_3d[(object_id, kp_id)])
                    # If we have 2D but no 3D, try to calculate it from camera pose
                    elif data.get('2d') is not None and camera_pose is not None and self.camera_intrinsic is not None:
                        point_2d = np.array(data['2d'])
                        rvec = np.array(camera_pose['rvec'])
                        tvec = np.array(camera_pose['tvec'])
                        # Use simple back-projection with estimated depth
                        # This is a fallback - ideally we should have triangulated 3D
                        fx = self.camera_intrinsic[0, 0]
                        fy = self.camera_intrinsic[1, 1]
                        cx = self.camera_intrinsic[0, 2]
                        cy = self.camera_intrinsic[1, 2]
                        x_norm = (point_2d[0] - cx) / fx
                        y_norm = (point_2d[1] - cy) / fy
                        # Estimate depth (use average distance to ArUco markers if available)
                        depth = 1.0
                        if hasattr(self, 'aruco_poses') and self.aruco_poses:
                            R, _ = cv2.Rodrigues(rvec)
                            distances = []
                            for marker_id, marker_pose in self.aruco_poses.items():
                                marker_tvec = np.array(marker_pose['tvec'])
                                marker_pos_world = marker_tvec
                                camera_pos_world = -R.T @ tvec
                                dist = np.linalg.norm(marker_pos_world - camera_pos_world)
                                distances.append(dist)
                            if distances:
                                depth = np.mean(distances)
                        # Back-project to 3D in camera frame
                        point_3d_cam = np.array([x_norm * depth, y_norm * depth, depth])
                        # Transform to world coordinates
                        R, _ = cv2.Rodrigues(rvec)
                        point_3d_world = R.T @ point_3d_cam - R.T @ tvec
                        keypoints_3d[kp_id] = point_3d_world
        
        # Also check calculated 3D keypoints
        for kp_key in self.calculated_3d.keys():
            if kp_key[0] == object_id:
                kp_id = kp_key[1]
                if kp_id not in keypoints_3d:
                    keypoints_3d[kp_id] = np.array(self.calculated_3d[kp_key])
        
        # Need at least 3 keypoints to check visibility
        if len(keypoints_3d) < 3:
            return True  # Skip objects with < 3 keypoints (they are considered visible)
        
        # Calculate object center (average of all keypoints)
        keypoint_positions = list(keypoints_3d.values())
        object_center = np.mean(keypoint_positions, axis=0)
        
        # Get camera position from camera pose
        if frame_id not in self.camera_poses:
            return True  # If no camera pose, assume visible
        
        camera_pose = self.camera_poses[frame_id]
        rvec = np.array(camera_pose['rvec'])
        tvec = np.array(camera_pose['tvec'])
        
        # Camera position in world frame: -R^T @ tvec
        R, _ = cv2.Rodrigues(rvec)
        camera_pos = -R.T @ tvec
        
        # Vector from camera to object center
        camera_to_center = object_center - camera_pos
        camera_to_center_norm = np.linalg.norm(camera_to_center)
        if camera_to_center_norm < 1e-6:
            return True  # Too close, assume visible
        camera_to_center = camera_to_center / camera_to_center_norm
        
        # Get three keypoints A, B, C where A < B < C by keypoint_id
        sorted_kp_ids = sorted(keypoints_3d.keys())
        if len(sorted_kp_ids) < 3:
            return True  # Should not happen, but safety check
        
        # Use first three keypoints
        kp_a_id = sorted_kp_ids[0]
        kp_b_id = sorted_kp_ids[1]
        kp_c_id = sorted_kp_ids[2]
        
        kp_a = keypoints_3d[kp_a_id]
        kp_b = keypoints_3d[kp_b_id]
        kp_c = keypoints_3d[kp_c_id]
        
        # Calculate vectors A->B and B->C
        vec_ab = kp_b - kp_a
        vec_bc = kp_c - kp_b
        
        # Calculate cross product of vec_ab and vec_bc
        cross_product = np.cross(vec_ab, vec_bc)
        cross_product_norm = np.linalg.norm(cross_product)
        if cross_product_norm < 1e-6:
            return True  # Vectors are parallel, assume visible
        cross_product = cross_product / cross_product_norm
        
        # Calculate angle between camera_to_center and cross_product
        dot_product = np.dot(camera_to_center, cross_product)
        # Clamp to [-1, 1] to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        # Object is invisible if angle > threshold
        visibility_threshold = DEFAULT_VISIBILITY_THRESHOLD
        is_visible = angle_deg <= visibility_threshold
        
        # Ensure Python bool (not numpy bool) for JSON serialization
        return bool(is_visible)
    
    def on_image_clicked(self, x, y):
        """Handle click on image - directly save keypoint"""
        print(f"[CLICK] Frame {self.current_frame}: Clicked at ({x}, {y})")
        
        # Get current IDs
        object_id = self.object_id_spin.value()
        keypoint_id = self.keypoint_id_spin.value()
        cls_id = self.cls_id_spin.value()
        
        # Emit signal to save keypoint
        self.keypoint_labeled.emit(self.current_frame, object_id, keypoint_id, [x, y], cls_id)
        
        # Update display
        self.image_label.add_keypoint(x, y, object_id, keypoint_id, self.current_frame, is_calculated=False, cls_id=cls_id)
        self.image_label.set_current_keypoint(None, None)  # Clear current keypoint
        
        print(f"[LABELING] Frame {self.current_frame}: Labeled Object {object_id}, Keypoint {keypoint_id} at ({x}, {y})")
        self.log_message(f"Labeled: Frame {self.current_frame}, Object {object_id}, Keypoint {keypoint_id} at ({x}, {y})")
        
        # Auto-increment keypoint_id (1 to max_keypoint_num, circular)
        # If wrapping from max_keypoint_num to 1, also increment object_id
        current_kp_id = self.keypoint_id_spin.value()
        if current_kp_id == DEFAULT_MAX_KEYPOINT_NUM:
            self.keypoint_id_spin.setValue(1)  # Wrap around to 1
            # Increment object_id when wrapping keypoint_id
            current_obj_id = self.object_id_spin.value()
            self.object_id_spin.setValue(current_obj_id + 1)
        else:
            self.keypoint_id_spin.setValue(current_kp_id + 1)
        
        # Check if we should triangulate and back-project (automatic)
        self.check_and_triangulate_keypoint(object_id, keypoint_id)
    
    def check_recommendations(self, x, y):
        """Check if we can recommend object/keypoint IDs based on existing labels"""
        if not self.labeling_data:
            self.recommendation_label.setText("")
            return
        
        # Find all existing keypoints with 3D locations
        existing_keypoints = []
        for frame_id, objects in self.labeling_data.items():
            for obj_id, keypoints in objects.items():
                for kp_id, data in keypoints.items():
                    if data.get('3d') is not None:
                        existing_keypoints.append({
                            'frame_id': frame_id,
                            'object_id': obj_id,
                            'keypoint_id': kp_id,
                            '3d': np.array(data['3d'])
                        })
        
        if len(existing_keypoints) < 2:
            self.recommendation_label.setText("")
            return
        
        # Project existing 3D keypoints to current frame and find closest
        min_dist = float('inf')
        best_match = None
        
        # Get camera pose for current frame
        if self.current_frame not in self.camera_poses:
            self.recommendation_label.setText("")
            return
        
        current_cam_pose = self.camera_poses[self.current_frame]
        current_rvec = np.array(current_cam_pose['rvec'])
        current_tvec = np.array(current_cam_pose['tvec'])
        
        for kp in existing_keypoints:
            if kp['frame_id'] == self.current_frame:
                continue
            
            # Project 3D point to current frame using camera pose
            pt_3d = kp['3d']
            
            # Project to current frame
            projected, _ = cv2.projectPoints(
                pt_3d.reshape(1, 3),
                current_rvec,
                current_tvec,
                self.camera_intrinsic,
                None
            )
            projected_2d = projected[0, 0]
            
            # Compute distance
            dist = np.sqrt((x - projected_2d[0])**2 + (y - projected_2d[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_match = kp
        
        from .constants import RECOMMENDATION_DISTANCE_THRESHOLD
        if best_match and min_dist < RECOMMENDATION_DISTANCE_THRESHOLD:
            self.recommendation_label.setText(
                f"Recommended: Object ID {best_match['object_id']}, "
                f"Keypoint ID {best_match['keypoint_id']} "
                f"(distance: {min_dist:.1f}px)"
            )
            self.object_id_spin.setValue(best_match['object_id'])
            self.keypoint_id_spin.setValue(best_match['keypoint_id'])
        else:
            self.recommendation_label.setText("")
    
    def on_done_clicked(self):
        """Handle Done button - now optional, keypoints are saved on click"""
        # This function is kept for backward compatibility but is no longer required
        # Keypoints are now saved directly on mouse click
        pass
    
    def on_calculate_triangulation_clicked(self):
        """Handle Calculate Triangulation button click - manually trigger triangulation for all keypoints"""
        print(f"\n[MANUAL TRIANGULATION] Starting manual triangulation calculation...")
        self.log_message("Calculating triangulation for all keypoints...")
        
        # Get all unique keypoint pairs
        keypoint_pairs = set()
        for frame_id, objects in self.labeling_data.items():
            for obj_id, keypoints in objects.items():
                for kp_id in keypoints.keys():
                    keypoint_pairs.add((obj_id, kp_id))
        
        print(f"[MANUAL TRIANGULATION] Found {len(keypoint_pairs)} unique keypoint pairs")
        
        triangulated_count = 0
        for obj_id, kp_id in keypoint_pairs:
            # Count observations
            obs_count = sum(1 for frame_id, objects in self.labeling_data.items()
                          if obj_id in objects and kp_id in objects[obj_id]
                          and objects[obj_id][kp_id].get('2d') is not None)
            
            if obs_count >= 3:
                print(f"[MANUAL TRIANGULATION] Triangulating Object {obj_id}, Keypoint {kp_id} ({obs_count} observations)")
                self.check_and_triangulate_keypoint(obj_id, kp_id)
                triangulated_count += 1
            else:
                print(f"[MANUAL TRIANGULATION] Skipping Object {obj_id}, Keypoint {kp_id} ({obs_count} observations, need 3+)")
        
        print(f"[MANUAL TRIANGULATION] Completed: {triangulated_count} keypoints triangulated")
        self.log_message(f"Triangulation complete: {triangulated_count} keypoints triangulated")
    
    def check_all_keypoints_for_triangulation(self):
        """Check all keypoints and triangulate those with >= 3 labels"""
        # Get all unique (object_id, keypoint_id) pairs
        keypoint_pairs = set()
        for frame_id, objects in self.labeling_data.items():
            for obj_id, keypoints in objects.items():
                for kp_id in keypoints.keys():
                    keypoint_pairs.add((obj_id, kp_id))
        
        # Check each keypoint
        for obj_id, kp_id in keypoint_pairs:
            self.check_and_triangulate_keypoint(obj_id, kp_id)
    
    def check_and_triangulate_keypoint(self, object_id, keypoint_id):
        """Check if keypoint has >= 3 labels, then triangulate and back-project to all frames"""
        print(f"\n[TRIANGULATION] Checking Object {object_id}, Keypoint {keypoint_id}...")
        
        # Collect all 2D observations of this keypoint
        observations = []  # List of (frame_id, point_2d)
        for frame_id, objects in self.labeling_data.items():
            if object_id in objects and keypoint_id in objects[object_id]:
                data = objects[object_id][keypoint_id]
                if data.get('2d') is not None:
                    pt_2d = np.array(data['2d'])
                    observations.append((frame_id, pt_2d))
                    print(f"[TRIANGULATION]   Observation from frame {frame_id}: ({pt_2d[0]:.2f}, {pt_2d[1]:.2f})")
        
        # Need at least 3 observations for triangulation
        if len(observations) < 3:
            if len(observations) > 0:
                print(f"[TRIANGULATION]   Insufficient observations: {len(observations)} (need 3)")
                self.log_message(f"Keypoint O{object_id}_K{keypoint_id}: {len(observations)} labels (need 3 for triangulation)")
            return
        
        print(f"[TRIANGULATION]   Starting triangulation with {len(observations)} observations...")
        self.log_message(f"Triangulating keypoint O{object_id}_K{keypoint_id} with {len(observations)} observations...")
        
        # Use triangulation to compute 3D location
        # Use first 3 observations for initial triangulation
        frame1, pt1 = observations[0]
        frame2, pt2 = observations[1]
        frame3, pt3 = observations[2]
        
        # Get camera poses
        if (frame1 not in self.camera_poses or 
            frame2 not in self.camera_poses or 
            frame3 not in self.camera_poses):
            print(f"[TRIANGULATION]   ERROR: Missing camera poses for frames {frame1}, {frame2}, or {frame3}")
            return
        
        print(f"[TRIANGULATION]   Using frames {frame1}, {frame2}, {frame3} for triangulation")
        
        rvec1 = np.array(self.camera_poses[frame1]['rvec'])
        tvec1 = np.array(self.camera_poses[frame1]['tvec'])
        rvec2 = np.array(self.camera_poses[frame2]['rvec'])
        tvec2 = np.array(self.camera_poses[frame2]['tvec'])
        rvec3 = np.array(self.camera_poses[frame3]['rvec'])
        tvec3 = np.array(self.camera_poses[frame3]['tvec'])
        
        print(f"[TRIANGULATION]   Frame {frame1} camera pose: rvec={rvec1}, tvec={tvec1}")
        print(f"[TRIANGULATION]   Frame {frame2} camera pose: rvec={rvec2}, tvec={tvec2}")
        print(f"[TRIANGULATION]   Frame {frame3} camera pose: rvec={rvec3}, tvec={tvec3}")
        
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        R3, _ = cv2.Rodrigues(rvec3)
        
        # Build projection matrices
        P1 = self.camera_intrinsic @ np.hstack([R1, tvec1.reshape(3, 1)])
        P2 = self.camera_intrinsic @ np.hstack([R2, tvec2.reshape(3, 1)])
        P3 = self.camera_intrinsic @ np.hstack([R3, tvec3.reshape(3, 1)])
        
        print(f"[TRIANGULATION]   Camera intrinsic matrix:")
        print(f"[TRIANGULATION]     {self.camera_intrinsic}")
        
        # Triangulate from first two views using shared utility
        print(f"[TRIANGULATION]   Triangulating from views {frame1}-{frame2}...")
        try:
            pt_3d_12 = triangulate_points(pt1, pt2, P1, P2)
            print(f"[TRIANGULATION]     Result: [{pt_3d_12[0]:.6f}, {pt_3d_12[1]:.6f}, {pt_3d_12[2]:.6f}]")
        except ValueError as e:
            print(f"[TRIANGULATION]     ERROR: {e}")
            return
        
        # Also triangulate from views 1-3 and 2-3, then average
        print(f"[TRIANGULATION]   Triangulating from views {frame1}-{frame3}...")
        try:
            pt_3d_13 = triangulate_points(pt1, pt3, P1, P3)
            print(f"[TRIANGULATION]     Result: [{pt_3d_13[0]:.6f}, {pt_3d_13[1]:.6f}, {pt_3d_13[2]:.6f}]")
        except ValueError as e:
            print(f"[TRIANGULATION]     ERROR: {e}")
            return
        
        print(f"[TRIANGULATION]   Triangulating from views {frame2}-{frame3}...")
        try:
            pt_3d_23 = triangulate_points(pt2, pt3, P2, P3)
            print(f"[TRIANGULATION]     Result: [{pt_3d_23[0]:.6f}, {pt_3d_23[1]:.6f}, {pt_3d_23[2]:.6f}]")
        except ValueError as e:
            print(f"[TRIANGULATION]     ERROR: {e}")
            return
        
        # Average the triangulated points
        pt_3d = (pt_3d_12 + pt_3d_13 + pt_3d_23) / 3.0
        
        # Calculate consistency
        from .constants import TRIANGULATION_CONSISTENCY_THRESHOLD
        diff_12_13 = np.linalg.norm(pt_3d_12 - pt_3d_13)
        diff_12_23 = np.linalg.norm(pt_3d_12 - pt_3d_23)
        diff_13_23 = np.linalg.norm(pt_3d_13 - pt_3d_23)
        max_diff = max(diff_12_13, diff_12_23, diff_13_23)
        print(f"[TRIANGULATION]   Consistency check:")
        print(f"[TRIANGULATION]     Max difference between pairs: {max_diff:.6f}")
        if max_diff > TRIANGULATION_CONSISTENCY_THRESHOLD:
            print(f"[TRIANGULATION]     WARNING: Large inconsistency detected!")
        
        # Store 3D location
        self.calculated_3d[(object_id, keypoint_id)] = pt_3d.tolist()
        
        # Back-project to all frames
        # Clear old calculated 2D locations for THIS keypoint only
        for frame_id in list(self.calculated_2d.keys()):
            if object_id in self.calculated_2d[frame_id]:
                if keypoint_id in self.calculated_2d[frame_id][object_id]:
                    del self.calculated_2d[frame_id][object_id][keypoint_id]
                    # Clean up empty dictionaries
                    if not self.calculated_2d[frame_id][object_id]:
                        del self.calculated_2d[frame_id][object_id]
                    if not self.calculated_2d[frame_id]:
                        del self.calculated_2d[frame_id]
        
        # Project to all frames
        print(f"[TRIANGULATION]   Projecting 3D point to all {len(self.images)} frames...")
        calculated_count = 0
        out_of_bounds_count = 0
        user_labeled_count = 0
        
        for frame_id in range(len(self.images)):
            # Validate frame_id before accessing
            if not validate_frame_id(frame_id, len(self.images)):
                continue
            
            if frame_id in self.camera_poses:
                camera_pose = self.camera_poses[frame_id]
                rvec = np.array(camera_pose['rvec'])
                tvec = np.array(camera_pose['tvec'])
                
                # Project 3D point to this frame
                projected, _ = cv2.projectPoints(
                    pt_3d.reshape(1, 3),
                    rvec,
                    tvec,
                    self.camera_intrinsic,
                    None
                )
                projected_2d = projected[0, 0]
                
                # Check if point is within image bounds
                if not validate_frame_id(frame_id, len(self.images)):
                    continue
                
                h, w = self.images[frame_id].shape[:2]
                if 0 <= projected_2d[0] < w and 0 <= projected_2d[1] < h:
                    # Store calculated 2D location (only if not user-labeled)
                    is_user_labeled = (frame_id in self.labeling_data and
                                      object_id in self.labeling_data[frame_id] and
                                      keypoint_id in self.labeling_data[frame_id][object_id])
                    
                    if not is_user_labeled:
                        if frame_id not in self.calculated_2d:
                            self.calculated_2d[frame_id] = {}
                        if object_id not in self.calculated_2d[frame_id]:
                            self.calculated_2d[frame_id][object_id] = {}
                        self.calculated_2d[frame_id][object_id][keypoint_id] = projected_2d.tolist()
                        calculated_count += 1
                    else:
                        user_labeled_count += 1
                else:
                    out_of_bounds_count += 1
        
        print(f"[TRIANGULATION]   Projection results: {calculated_count} calculated, {user_labeled_count} user-labeled, {out_of_bounds_count} out of bounds")
        
        # Output to terminal for debugging
        print(f"\n=== Triangulated Keypoint ===")
        print(f"Object ID: {object_id}, Keypoint ID: {keypoint_id}")
        print(f"3D Location: [{pt_3d[0]:.6f}, {pt_3d[1]:.6f}, {pt_3d[2]:.6f}]")
        print(f"Number of observations: {len(observations)}")
        print(f"Projected to {calculated_count} frames")
        print(f"Triangulation results:")
        print(f"  From views 1-2: [{pt_3d_12[0]:.6f}, {pt_3d_12[1]:.6f}, {pt_3d_12[2]:.6f}]")
        print(f"  From views 1-3: [{pt_3d_13[0]:.6f}, {pt_3d_13[1]:.6f}, {pt_3d_13[2]:.6f}]")
        print(f"  From views 2-3: [{pt_3d_23[0]:.6f}, {pt_3d_23[1]:.6f}, {pt_3d_23[2]:.6f}]")
        print(f"  Average: [{pt_3d[0]:.6f}, {pt_3d[1]:.6f}, {pt_3d[2]:.6f}]")
        print(f"=============================\n")
        
        # Update display for current frame
        self.update_frame_display()
        
        self.log_message(f"Triangulated keypoint: Object {object_id}, Keypoint {keypoint_id} "
                        f"(3D: [{pt_3d[0]:.3f}, {pt_3d[1]:.3f}, {pt_3d[2]:.3f}], "
                        f"projected to {calculated_count} frames)")
    
    # Note: triangulate_points is now imported from utils module to avoid duplication
    
    def log_message(self, message):
        self.log_text.append(message)
    
    def reset_session(self):
        """Reset the labeling session - clear all data"""
        # Clear all labeling data
        self.images = []
        self.image_paths = []
        self.aruco_poses = {}
        self.camera_poses = {}
        self.camera_intrinsic = None
        self.frame_detections = {}
        self.marker_size = 0.05
        self.labeling_data = {}
        self.calculated_2d = {}
        self.calculated_3d = {}
        self.calculated_visibility = {}  # {(frame_id, object_id, keypoint_id): bool} - visibility for triangulated keypoints loaded from JSON
        self.current_frame = 0
        self.is_setup = False
        self.saved_json_path = None
        self.saved_yolo_path = None
        
        # Reset UI
        self.image_label.set_image(None)
        self.image_label.clear_keypoints()
        self.timeline_slider.setMaximum(0)
        self.frame_label.setText("Frame: 0 / 0")
        self.save_label_btn.setEnabled(False)
        self.save_yolo_btn.setEnabled(False)
        self.object_id_spin.setValue(1)
        self.keypoint_id_spin.setValue(1)
        self.cls_id_spin.setValue(1)
        
        # Clear log or add separator
        self.log_message("\n" + "="*60)
        self.log_message("Previous session cleared. Starting new session...")
        self.log_message("="*60 + "\n")

