"""
Main window for the 3D Keypoints Labeling Application
"""

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                                QHBoxLayout, QMessageBox, QTabWidget,
                                QDialog, QDialogButtonBox, QLabel, QSpinBox,
                                QApplication, QProgressDialog)
from PySide6.QtCore import Qt
import os
import sys
import json
import shutil
import re
import numpy as np
import cv2
import tempfile
from pathlib import Path
from collections import defaultdict
from .aruco_processor import ArucoProcessor
from .labeling_tab import LabelingTab
from .trajectory_tab import TrajectoryTab

# Import default configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import default_cfg
    RECOMMEND_IMAGE_NUM = default_cfg.recommend_image_num
    DEFAULT_DOWNSAMPLE_RATIO = default_cfg.downsample_ratio
    MAX_KEYPOINT_NUM = default_cfg.max_keypoint_num
except ImportError:
    RECOMMEND_IMAGE_NUM = 200  # Fallback default
    DEFAULT_DOWNSAMPLE_RATIO = 15  # Fallback default
    MAX_KEYPOINT_NUM = 4  # Fallback default

# Import utilities
try:
    from .utils import (
        safe_read_json, safe_write_json, validate_frame_id, 
        get_image_name_safe, safe_get_nested_dict
    )
    USE_UTILS = True
except ImportError:
    USE_UTILS = False
    # Fallback implementations
    def validate_frame_id(frame_id: int, num_frames: int) -> bool:
        return 0 <= frame_id < num_frames
    
    def get_image_name_safe(image_paths, frame_id: int) -> str:
        if not validate_frame_id(frame_id, len(image_paths)):
            raise IndexError(f"Frame ID {frame_id} out of range")
        return os.path.basename(image_paths[frame_id])
    
    def safe_get_nested_dict(data, *keys, default=None):
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Keypoints Labeling Application")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage
        self.image_dir = None
        self.original_image_dir = None  # Store original image directory before downsampling
        self.temp_downsample_dir = None  # Temporary directory for downsampled images
        self.temp_to_original_path_map = {}  # Map from temp image paths to original image paths
        self.aruco_params = {}
        self.camera_intrinsic = None
        self.aruco_poses = {}
        self.camera_poses = {}  # Per-frame camera poses
        self.frame_aruco_count = {}  # {frame_id: count} - number of ArUco markers per frame
        self.images = []
        self.image_paths = []
        self.labeling_data = {}  # {frame_id: {object_id: {keypoint_id: {'2d': [x, y], '3d': [x, y, z]}}}}
        self.loaded_json_path = None
        self.saved_json_path = None  # Track if JSON has been saved during this session
        self.saved_yolo_path = None  # Track if YOLO format has been saved during this session
        
        # Central widget with tabs
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        layout = QVBoxLayout(self.central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Tab 1: Labeling (includes parameter widget on the right)
        self.labeling_tab = LabelingTab()
        self.labeling_tab.keypoint_labeled.connect(self.on_keypoint_labeled)
        self.labeling_tab.keypoint_removed.connect(self.on_keypoint_removed)
        self.labeling_tab.done_clicked.connect(self.on_done_clicked)
        self.labeling_tab.save_label_clicked.connect(self.on_save_label_clicked)
        self.labeling_tab.save_yolo_clicked.connect(self.on_save_yolo_clicked)
        self.labeling_tab.parameters_ready.connect(self.on_parameters_ready)
        self.labeling_tab.exit_requested.connect(self.on_exit_requested)
        self.tab_widget.addTab(self.labeling_tab, "Labeling")
        
        # Tab 2: Trajectory visualization
        self.trajectory_tab = TrajectoryTab()
        self.tab_widget.addTab(self.trajectory_tab, "Trajectory")
        
        # Progress dialog for ArUco processing
        self.progress_dialog = None
        
    def on_parameters_ready(self, params):
        """Handle parameters from Tab 1"""
        # Clear previous session data in main window
        self.image_dir = None
        self.aruco_params = {}
        self.camera_intrinsic = None
        self.aruco_poses = {}
        self.camera_poses = {}
        self.frame_aruco_count = {}
        self.images = []
        self.image_paths = []
        self.labeling_data = {}
        self.loaded_json_path = None
        self.saved_json_path = None
        self.saved_yolo_path = None
        
        # Store params for later use
        self._last_params = params
        
        # Set new session parameters
        self.image_dir = params['image_dir']
        self.aruco_params = {
            'size': params['aruco_size'],
            'physical_length': params['physical_length'],
            'dict_type': params['dict_type']
        }
        self.camera_intrinsic = params.get('camera_intrinsic')
        self.loaded_json_path = params.get('json_path')
        
        # Load existing labeling data if JSON path provided
        load_result = None
        if self.loaded_json_path and os.path.exists(self.loaded_json_path):
            load_result = self.load_labeling_data(self.loaded_json_path)
        
        # Check if metadata was loaded (ArUco poses and camera poses)
        if load_result and load_result.get('has_metadata', False):
            # Skip ArUco processing, just load images
            self.labeling_tab.log_message("ArUco poses and camera poses found in JSON file, skipping ArUco processing...")
            self.load_images_without_aruco()
        else:
            # Start ArUco processing
            self.process_aruco_markers()
    
    def remap_poses_by_image_name(self):
        """Remap camera poses and frame_aruco_count from old frame_id to new frame_id based on image_name"""
        if not hasattr(self, '_loaded_camera_poses') or not self._loaded_camera_poses:
            return
        
        # Get mapping from image_name to old frame_id from metadata
        image_name_to_old_frame_id = {}
        try:
            with open(self.loaded_json_path, 'r') as f:
                file_data = json.load(f)
            
            if isinstance(file_data, dict) and 'metadata' in file_data:
                metadata = file_data['metadata']
                if 'image_name_to_frame_id' in metadata:
                    # Use the saved mapping
                    image_name_to_old_frame_id = {k: int(v) for k, v in metadata['image_name_to_frame_id'].items()}
        except Exception as e:
            self.labeling_tab.log_message(f"Warning: Could not load image_name mapping: {str(e)}")
        
        # Build mapping from image_name to new frame_id (current image_paths)
        image_name_to_new_frame_id = {}
        for idx, path in enumerate(self.image_paths):
            image_name = os.path.basename(path)
            image_name_to_new_frame_id[image_name] = idx
        
        # Remap camera poses using image_name as the key
        if image_name_to_old_frame_id:
            remapped_camera_poses = {}
            remapped_frame_aruco_count = {}
            
            for image_name, old_frame_id in image_name_to_old_frame_id.items():
                new_frame_id = image_name_to_new_frame_id.get(image_name)
                if new_frame_id is not None:
                    # Remap camera pose
                    if old_frame_id in self._loaded_camera_poses:
                        remapped_camera_poses[new_frame_id] = self._loaded_camera_poses[old_frame_id]
                    
                    # Remap frame_aruco_count
                    if old_frame_id in self._loaded_frame_aruco_count:
                        remapped_frame_aruco_count[new_frame_id] = self._loaded_frame_aruco_count[old_frame_id]
            
            self.camera_poses = remapped_camera_poses
            self.frame_aruco_count = remapped_frame_aruco_count
            self.labeling_tab.log_message(f"Remapped {len(remapped_camera_poses)} camera poses based on image names")
        else:
            # Fallback: assume frame_ids correspond to sorted image order
            # This is risky but better than nothing
            sorted_old_frame_ids = sorted(self._loaded_camera_poses.keys())
            if len(sorted_old_frame_ids) <= len(self.image_paths):
                self.labeling_tab.log_message("Warning: Could not find image_name mapping, using frame order assumption")
                remapped_camera_poses = {}
                remapped_frame_aruco_count = {}
                for idx, old_frame_id in enumerate(sorted_old_frame_ids):
                    if idx < len(self.image_paths):
                        remapped_camera_poses[idx] = self._loaded_camera_poses[old_frame_id]
                        if old_frame_id in self._loaded_frame_aruco_count:
                            remapped_frame_aruco_count[idx] = self._loaded_frame_aruco_count[old_frame_id]
                self.camera_poses = remapped_camera_poses
                self.frame_aruco_count = remapped_frame_aruco_count
    
    def load_images_without_aruco(self):
        """Load images without ArUco processing (when poses are loaded from JSON)"""
        import glob
        
        self.labeling_tab.log_message("Loading images...")
        
        # Load images
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")) +
                            glob.glob(os.path.join(self.image_dir, "*.png")) +
                            glob.glob(os.path.join(self.image_dir, "*.jpeg")))
        
        if not image_paths:
            QMessageBox.warning(self, "Error", "No images found in directory")
            return
        
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
        
        if not images:
            QMessageBox.warning(self, "Error", "Failed to load images")
            return
        
        self.images = images
        self.image_paths = image_paths
        
        # Set marker size from params (needed for display)
        if hasattr(self, '_last_params') and 'aruco_params' in self._last_params:
            # Use default marker size
            self.marker_size = 0.05
        else:
            self.marker_size = 0.05
        
        # Initialize frame_detections as empty (not needed when loading from JSON)
        self.frame_detections = {}
        
        self.labeling_tab.log_message(f"Loaded {len(images)} images")
        
        # Reload JSON file now that images are available (to convert image_name to frame_id)
        if self.loaded_json_path and os.path.exists(self.loaded_json_path):
            self.labeling_tab.log_message("Reloading JSON file now that images are available...")
            self.load_labeling_data(self.loaded_json_path, merge=True)
            
            # Remap poses based on image_name correspondence
            self.remap_poses_by_image_name()
        
        # Initialize labeling tab with loaded data
        self.labeling_tab.setup(
            self.images,
            self.image_paths,
            self.aruco_poses,
            self.camera_poses,
            self.camera_intrinsic,
            self.labeling_data,
            self.frame_aruco_count,
            self.frame_detections,
            self.marker_size
        )
        
        # Update trajectory visualization
        self.trajectory_tab.update_visualization(
            self.aruco_poses,
            self.camera_poses,
            self.camera_intrinsic
        )
        
        # Switch to trajectory tab to show results
        self.tab_widget.setCurrentIndex(1)
        
        # Re-enable start button
        self.labeling_tab.start_btn.setEnabled(True)
        self.labeling_tab.log_message("Loading complete! You can now continue labeling or import another video/directory.")
    
    def _show_downsample_dialog(self, num_images):
        """Show dialog to get downsample ratio from user for large image sets"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Large Image Set Warning")
        dialog.setMinimumWidth(450)
        
        layout = QVBoxLayout(dialog)
        
        # Message label
        message = QLabel(
            f"The number of images is {num_images}, which exceeds the recommended number ({RECOMMEND_IMAGE_NUM}).\n\n"
            f"Bundle adjustment may take a significant amount of time with this many images.\n\n"
            f"Would you like to downsample the images?\n"
            f"Please specify the downsample ratio (use every Nth image):"
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        
        # Downsample input
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Downsample ratio:"))
        downsample_spin = QSpinBox()
        downsample_spin.setMinimum(1)
        downsample_spin.setMaximum(1000)
        # Calculate a reasonable default downsample ratio to get close to RECOMMEND_IMAGE_NUM
        suggested_ratio = max(1, (num_images + RECOMMEND_IMAGE_NUM - 1) // RECOMMEND_IMAGE_NUM)
        downsample_spin.setValue(min(suggested_ratio, DEFAULT_DOWNSAMPLE_RATIO))
        input_layout.addWidget(downsample_spin)
        
        # Show estimated number of images after downsampling
        def update_estimate():
            ratio = downsample_spin.value()
            estimated = (num_images + ratio - 1) // ratio  # Ceiling division
            estimate_label.setText(f"(Estimated: ~{estimated} images)")
        
        estimate_label = QLabel()
        update_estimate()
        downsample_spin.valueChanged.connect(update_estimate)
        input_layout.addWidget(estimate_label)
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
    
    def _create_downsampled_image_dir(self, image_paths, downsample_ratio):
        """Create a temporary directory with downsampled images (symlinks)
        
        Note: Consider using tempfile.TemporaryDirectory context manager in future
        """
        # Create temporary directory
        try:
            temp_dir = tempfile.mkdtemp(prefix='labelkeypoints3d_downsample_')
        except OSError as e:
            raise OSError(f"Failed to create temporary directory: {e}")
        
        self.temp_downsample_dir = temp_dir
        self.temp_to_original_path_map = {}
        
        # Filter image paths by downsample ratio
        filtered_paths = image_paths[::downsample_ratio]
        
        # Create symlinks to filtered images and build mapping
        for img_path in filtered_paths:
            img_name = os.path.basename(img_path)
            link_path = os.path.join(temp_dir, img_name)
            try:
                os.symlink(img_path, link_path)
            except OSError:
                # If symlink fails (e.g., on Windows without admin), copy the file
                shutil.copy2(img_path, link_path)
            
            # Store mapping from temp path to original path
            self.temp_to_original_path_map[link_path] = img_path
        
        return temp_dir, len(filtered_paths)
    
    def process_aruco_markers(self):
        """Process ArUco markers in background thread"""
        self.labeling_tab.log_message("Starting ArUco marker detection...")
        
        # Count images before processing to show warning if needed
        import glob
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")) +
                            glob.glob(os.path.join(self.image_dir, "*.png")) +
                            glob.glob(os.path.join(self.image_dir, "*.jpeg")))
        num_images = len(image_paths)
        
        # Check if number of images exceeds recommended limit (before camera pose calculation)
        if num_images > RECOMMEND_IMAGE_NUM:
            downsample_ratio = self._show_downsample_dialog(num_images)
            
            if downsample_ratio is not None:
                # User wants to downsample
                self.labeling_tab.log_message(f"Downsampling images with ratio {downsample_ratio}...")
                self.original_image_dir = self.image_dir
                temp_dir, filtered_count = self._create_downsampled_image_dir(image_paths, downsample_ratio)
                self.image_dir = temp_dir
                self.labeling_tab.log_message(f"Using {filtered_count} images (downsampled from {num_images})")
            else:
                # User cancelled or chose not to downsample
                self.labeling_tab.log_message(f"Proceeding with all {num_images} images (may take longer)")
        
        # Show progress dialog with progress bar
        self.progress_dialog = QProgressDialog("", None, 0, 100, self)
        self.progress_dialog.setWindowTitle("Processing")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setCancelButton(None)  # No cancel button
        self.progress_dialog.setMinimumDuration(0)  # Show immediately
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        QApplication.processEvents()  # Ensure dialog is shown
        
        # Get enable_ba from stored params
        enable_ba = False
        if hasattr(self, '_last_params') and 'enable_ba' in self._last_params:
            enable_ba = self._last_params['enable_ba']
        
        # Create processor
        self.aruco_processor = ArucoProcessor(
            self.image_dir,
            self.aruco_params,
            self.camera_intrinsic,
            enable_ba=enable_ba
        )
        # Connect progress signal to update dialog message
        self.aruco_processor.progress_signal.connect(self._update_progress_message)
        self.aruco_processor.finished_signal.connect(self.on_aruco_processing_complete)
        self.aruco_processor.start()
    
    def _update_progress_message(self, message, current=0, total=0):
        """Update progress dialog message with standardized format"""
        if self.progress_dialog:
            # Extract task name from message
            task_name = "Processing"
            if message:
                msg_lower = message.lower()
                # Identify specific tasks from message content
                # Check for "loading images" FIRST to distinguish it from detection
                if "loading images" in msg_lower:
                    task_name = "Loading Images"
                elif "bundle adjustment" in msg_lower or "performing bundle adjustment" in msg_lower:
                    task_name = "Bundle Adjustment"
                elif "optimizing" in msg_lower or "iter" in msg_lower:
                    task_name = "Bundle Adjustment"
                elif "detecting aruco" in msg_lower or ("detected" in msg_lower and "aruco" in msg_lower):
                    task_name = "Detecting ArUco"
                elif "loaded" in msg_lower and "images" in msg_lower:
                    # "Loaded X images" - transition from loading to detection
                    task_name = "Loading Images"
                elif "filtered" in msg_lower:
                    task_name = "Detecting ArUco"  # Filtering is part of ArUco detection
                elif "..." in message:
                    task_name = message.split("...")[0].strip()
                elif "(" in message:
                    task_name = message.split("(")[0].strip()
                elif ":" in message:
                    task_name = message.split(":")[0].strip()
                else:
                    task_name = message.strip()
            
            # Update window title with task name
            self.progress_dialog.setWindowTitle(task_name)
            
            # Format label text as "Task Name: current / total" if we have valid values
            if total > 0:
                # Extract reprojection error from message if it's bundle adjustment optimization
                error_text = ""
                if message and ("error" in message.lower() or "optimizing" in message.lower()):
                    # Try to extract error value from message (format: "...error = X.XX pixels")
                    error_match = re.search(r'error\s*=\s*([\d.]+)\s*pixels', message, re.IGNORECASE)
                    if error_match:
                        error_value = float(error_match.group(1))
                        error_text = f" (error: {error_value:.2f} px)"
                label_text = f"{task_name}: {current} / {total}{error_text}"
                self.progress_dialog.setLabelText(label_text)
                # Update progress bar
                progress_pct = int(100 * current / total)
                self.progress_dialog.setValue(progress_pct)
                self.progress_dialog.setRange(0, 100)
            else:
                # Indeterminate progress (spinner mode) - show task name and message
                label_text = f"{task_name}: {message}" if message else f"{task_name}..."
                self.progress_dialog.setLabelText(label_text)
                self.progress_dialog.setRange(0, 0)
            QApplication.processEvents()  # Update UI
    
    def on_aruco_processing_complete(self, result):
        """Handle completion of ArUco processing"""
        # Hide progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if result['success']:
            self.aruco_poses = result['aruco_poses']
            self.camera_poses = result.get('camera_poses', {})
            self.frame_aruco_count = result.get('frame_aruco_count', {})
            self.frame_detections = result.get('frame_detections', {})
            self.marker_size = result.get('marker_size', 0.05)
            self.images = result['images']
            self.image_paths = result['image_paths']
            self.camera_intrinsic = result['camera_intrinsic']
            
            # If we used a temporary directory for downsampling, map image paths back to originals
            if self.temp_to_original_path_map:
                mapped_paths = []
                for temp_path in self.image_paths:
                    # Try to find original path by matching image name
                    img_name = os.path.basename(temp_path)
                    original_path = None
                    for orig_path in self.temp_to_original_path_map.values():
                        if os.path.basename(orig_path) == img_name:
                            original_path = orig_path
                            break
                    if original_path:
                        mapped_paths.append(original_path)
                    else:
                        # Fallback: use temp path if mapping not found
                        mapped_paths.append(temp_path)
                self.image_paths = mapped_paths
                
                # Restore original image directory
                if self.original_image_dir:
                    self.image_dir = self.original_image_dir
                    self.original_image_dir = None
                
                # Clean up temporary directory
                if self.temp_downsample_dir and os.path.exists(self.temp_downsample_dir):
                    try:
                        shutil.rmtree(self.temp_downsample_dir)
                        self.labeling_tab.log_message(f"Cleaned up temporary downsampling directory")
                    except Exception as e:
                        self.labeling_tab.log_message(f"Warning: Could not clean up temporary directory: {str(e)}")
                    self.temp_downsample_dir = None
                    self.temp_to_original_path_map = {}
            
            # Reload JSON file if it was loaded before images were available
            # This ensures entries with 'image_name' can be properly converted to frame_id
            # Only reload if metadata was not loaded (i.e., we did ArUco processing)
            if self.loaded_json_path and os.path.exists(self.loaded_json_path):
                # Check if we need to reload (only if we didn't load metadata earlier)
                if not (hasattr(self, '_loaded_metadata') and self._loaded_metadata):
                    self.labeling_tab.log_message("Reloading JSON file now that images are available...")
                    self.load_labeling_data(self.loaded_json_path, merge=True)
            
            self.labeling_tab.log_message("ArUco processing completed successfully!")
            self.labeling_tab.log_message(f"Detected {len(self.aruco_poses)} ArUco marker poses")
            self.labeling_tab.log_message(f"Computed {len(self.camera_poses)} camera poses")
            
            # Output camera intrinsics to terminal
            print(f"\n=== Camera Intrinsics (After Bundle Adjustment) ===")
            K = self.camera_intrinsic
            print(f"fx: {K[0, 0]:.6f}")
            print(f"fy: {K[1, 1]:.6f}")
            print(f"cx: {K[0, 2]:.6f}")
            print(f"cy: {K[1, 2]:.6f}")
            print(f"Matrix:")
            print(f"  [{K[0, 0]:.6f}   0   {K[0, 2]:.6f}]")
            print(f"  [  0   {K[1, 1]:.6f}   {K[1, 2]:.6f}]")
            print(f"  [  0     0     1   ]")
            print(f"==================================================\n")
            
            # Initialize labeling tab
            self.labeling_tab.setup(
                self.images,
                self.image_paths,
                self.aruco_poses,
                self.camera_poses,
                self.camera_intrinsic,
                self.labeling_data,
                self.frame_aruco_count,
                self.frame_detections,
                self.marker_size
            )
            
            # Update trajectory visualization
            self.trajectory_tab.update_visualization(
                self.aruco_poses,
                self.camera_poses,
                self.camera_intrinsic
            )
            
            # Switch to trajectory tab to show results
            self.tab_widget.setCurrentIndex(1)
            
            # Re-enable start button to allow starting a new session
            self.labeling_tab.start_btn.setEnabled(True)
            self.labeling_tab.log_message("Processing complete! You can now import another video/directory to start a new labeling session.")
        else:
            QMessageBox.critical(self, "Error", result.get('error', 'Unknown error during ArUco processing'))
            # Re-enable start button even on error so user can retry
            self.labeling_tab.start_btn.setEnabled(True)
    
    def on_keypoint_labeled(self, frame_id, object_id, keypoint_id, point_2d, cls_id):
        """Handle keypoint labeling"""
        if frame_id not in self.labeling_data:
            self.labeling_data[frame_id] = {}
        if object_id not in self.labeling_data[frame_id]:
            self.labeling_data[frame_id][object_id] = {}
        
        self.labeling_data[frame_id][object_id][keypoint_id] = {
            '2d': point_2d,
            '3d': None,  # Will be computed during optimization
            'cls_id': cls_id
        }
    
    def on_keypoint_removed(self, frame_id, object_id, keypoint_id, x, y, is_calculated):
        """Handle keypoint removal"""
        # Data is already removed in labeling_tab, just update main window
        pass
    
    def on_done_clicked(self):
        """Handle Done button click - optimize and save"""
        if not self.labeling_data:
            QMessageBox.warning(self, "Warning", "No keypoints labeled yet!")
            return
        
        self.labeling_tab.log_message("Optimizing keypoint 3D locations...")
        
        # Optimize keypoints
        from .keypoint_optimizer import KeypointOptimizer
        optimizer = KeypointOptimizer(
            self.labeling_data,
            self.camera_poses,
            self.camera_intrinsic
        )
        optimized_data = optimizer.optimize()
        
        # Save to JSON
        output_path = self.save_labeling_data(optimized_data)
        self.labeling_tab.log_message(f"Labeling data saved to: {output_path}")
        QMessageBox.information(self, "Success", f"Labeling data saved to:\n{output_path}")
    
    def load_labeling_data(self, json_path, merge=False):
        """Load labeling data from JSON file with improved error handling
        
        Args:
            json_path: Path to JSON file
            merge: If True, merge with existing data instead of clearing it
        
        Returns:
            dict with 'has_metadata' key indicating if ArUco/camera poses were loaded
        """
        try:
            if USE_UTILS:
                file_data = safe_read_json(json_path)
            else:
                with open(json_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
            
            # Check if new format (with metadata) or old format (just keypoints array)
            if isinstance(file_data, dict) and 'metadata' in file_data:
                # New format with metadata
                metadata = file_data.get('metadata', {})
                data = file_data.get('keypoints', [])
                
                # Load metadata if available
                has_metadata = False
                loaded_camera_poses = {}
                loaded_aruco_poses = {}
                loaded_frame_aruco_count = {}
                
                if metadata:
                    # Load camera poses (with old frame_id keys)
                    if 'camera_poses' in metadata and metadata['camera_poses']:
                        loaded_camera_poses = {int(k): {'rvec': np.array(v['rvec']), 'tvec': np.array(v['tvec'])}
                                              for k, v in metadata['camera_poses'].items()}
                        has_metadata = True
                    
                    # Load ArUco poses (these don't depend on frame_id, so keep as is)
                    if 'aruco_poses' in metadata and metadata['aruco_poses']:
                        loaded_aruco_poses = {int(k): {'rvec': np.array(v['rvec']), 'tvec': np.array(v['tvec'])}
                                             for k, v in metadata['aruco_poses'].items()}
                        has_metadata = True
                    
                    # Load frame ArUco count (with old frame_id keys)
                    if 'frame_aruco_count' in metadata:
                        loaded_frame_aruco_count = {int(k): v for k, v in metadata['frame_aruco_count'].items()}
                    
                    # Load camera intrinsic
                    if 'camera_intrinsic' in metadata and metadata['camera_intrinsic']:
                        self.camera_intrinsic = np.array(metadata['camera_intrinsic'])
                
                # Store flag that metadata was loaded
                self._loaded_metadata = has_metadata
                self._loaded_camera_poses = loaded_camera_poses
                self._loaded_frame_aruco_count = loaded_frame_aruco_count
                self.aruco_poses = loaded_aruco_poses
            else:
                # Old format - just keypoints array
                data = file_data if isinstance(file_data, list) else []
                has_metadata = False
                self._loaded_metadata = False
            
            # Convert to internal format
            if not merge:
                self.labeling_data = {}
                self.labeling_tab.calculated_2d = {}
                self.labeling_tab.calculated_3d = {}
                # Store visibility for triangulated keypoints loaded from JSON
                if not hasattr(self.labeling_tab, 'calculated_visibility'):
                    self.labeling_tab.calculated_visibility = {}
            else:
                # Ensure dictionaries exist when merging
                if not hasattr(self.labeling_tab, 'calculated_2d') or self.labeling_tab.calculated_2d is None:
                    self.labeling_tab.calculated_2d = {}
                if not hasattr(self.labeling_tab, 'calculated_3d') or self.labeling_tab.calculated_3d is None:
                    self.labeling_tab.calculated_3d = {}
                if not hasattr(self.labeling_tab, 'calculated_visibility'):
                    self.labeling_tab.calculated_visibility = {}
            
            for item in data:
                # Support both old format (frame_id) and new format (image_name)
                if 'image_name' in item:
                    # New format: convert image_name to frame_id
                    image_name = item['image_name']
                    if self.image_paths:
                        # Find frame_id by matching image name
                        frame_id = None
                        for idx, path in enumerate(self.image_paths):
                            if os.path.basename(path) == image_name:
                                frame_id = idx
                                break
                        if frame_id is None:
                            # Image not found, skip this entry
                            continue
                    else:
                        # image_paths not loaded yet, skip for now
                        # This should not happen in normal flow, but handle gracefully
                        continue
                elif 'frame_id' in item:
                    # Old format: use frame_id directly (backward compatibility)
                    frame_id = item['frame_id']
                else:
                    # Neither key present, skip this entry
                    continue
                
                object_id = item['object_id']
                keypoint_id = item['keypoint_id']
                is_manual = item.get('is_manual', 1)  # Default to 1 (manual) if not present
                
                if is_manual == 1:
                    # User-labeled keypoint
                    if frame_id not in self.labeling_data:
                        self.labeling_data[frame_id] = {}
                    if object_id not in self.labeling_data[frame_id]:
                        self.labeling_data[frame_id][object_id] = {}
                    
                    self.labeling_data[frame_id][object_id][keypoint_id] = {
                        '2d': item['keypoint_2d'],
                        '3d': item['keypoint_3d'],
                        'cls_id': item.get('cls_id', 1),  # Default to 1 if not present (backward compatibility)
                        'visibility': item.get('visibility')  # Store visibility if present in JSON
                    }
                else:
                    # Triangulated keypoint
                    if frame_id not in self.labeling_tab.calculated_2d:
                        self.labeling_tab.calculated_2d[frame_id] = {}
                    if object_id not in self.labeling_tab.calculated_2d[frame_id]:
                        self.labeling_tab.calculated_2d[frame_id][object_id] = {}
                    
                    self.labeling_tab.calculated_2d[frame_id][object_id][keypoint_id] = item['keypoint_2d']
                    
                    # Also store 3D location
                    if item.get('keypoint_3d'):
                        self.labeling_tab.calculated_3d[(object_id, keypoint_id)] = item['keypoint_3d']
                    
                    # Store visibility if present in JSON
                    if 'visibility' in item and item['visibility'] is not None:
                        if not hasattr(self.labeling_tab, 'calculated_visibility'):
                            self.labeling_tab.calculated_visibility = {}
                        visibility_key = (frame_id, object_id, keypoint_id)
                        self.labeling_tab.calculated_visibility[visibility_key] = item['visibility']
            
            self.labeling_tab.log_message(f"Loaded {len(data)} keypoint labels from {json_path}")
            if has_metadata:
                self.labeling_tab.log_message(f"Loaded ArUco poses and camera poses from JSON file")
            # Update display to show loaded keypoints
            if self.labeling_tab.is_setup:
                self.labeling_tab.update_frame_display()
            
            return {'has_metadata': has_metadata}
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to load JSON file: {str(e)}")
            return {'has_metadata': False}
    
    def save_labeling_data(self, optimized_data):
        """Save labeling data to JSON file"""
        # Convert to output format
        output_data = []
        for frame_id, objects in optimized_data.items():
            # Get image name from frame_id with validation
            try:
                image_name = get_image_name_safe(self.image_paths, frame_id)
            except IndexError:
                image_name = f"frame_{frame_id}.jpg"
            
            # Track which objects are visible (to avoid checking multiple times)
            visible_objects = {}  # {object_id: bool}
            
            for object_id, keypoints in objects.items():
                # Check visibility for this object (only if checkbox is enabled)
                if self.labeling_tab.check_visibility_checkbox.isChecked():
                    if object_id not in visible_objects:
                        visible_objects[object_id] = self.labeling_tab.check_object_visibility(frame_id, object_id)
                else:
                    # If visibility check is disabled, consider all objects visible
                    visible_objects[object_id] = True
                
                for keypoint_id, data in keypoints.items():
                    # Get visibility status: use stored value from JSON if present, otherwise calculate
                    if 'visibility' in data and data['visibility'] is not None:
                        is_visible = bool(data['visibility'])
                    else:
                        # Calculate visibility if not stored
                        is_visible = bool(visible_objects.get(object_id, True))
                    
                    output_data.append({
                        'image_name': image_name,
                        'object_id': object_id,
                        'keypoint_id': keypoint_id,
                        'keypoint_3d': data['3d'],
                        'keypoint_2d': data['2d'],
                        'cls_id': data.get('cls_id', 1),  # Default to 1 if not present (backward compatibility)
                        'visibility': is_visible  # Visibility status
                    })
        
        # Determine output path
        if self.loaded_json_path:
            output_path = self.loaded_json_path
        else:
            output_path = os.path.join(self.image_dir, 'labeling_results.json')
        
        # Create image_name to frame_id mapping for pose remapping
        image_name_to_frame_id = {}
        for frame_id in range(len(self.image_paths)):
            try:
                image_name = get_image_name_safe(self.image_paths, frame_id)
                image_name_to_frame_id[image_name] = frame_id
            except IndexError:
                continue  # Skip invalid frame IDs
        
        # Filter frame_aruco_count to only include frames that have camera poses (i.e., not filtered out)
        filtered_frame_aruco_count = {
            str(k): v for k, v in self.frame_aruco_count.items() 
            if k in self.camera_poses
        }
        
        # Create output structure with metadata
        output_structure = {
            'metadata': {
                'aruco_poses': {str(k): {'rvec': v['rvec'].tolist() if isinstance(v['rvec'], np.ndarray) else v['rvec'],
                                        'tvec': v['tvec'].tolist() if isinstance(v['tvec'], np.ndarray) else v['tvec']}
                               for k, v in self.aruco_poses.items()},
                'camera_poses': {str(k): {'rvec': v['rvec'].tolist() if isinstance(v['rvec'], np.ndarray) else v['rvec'],
                                        'tvec': v['tvec'].tolist() if isinstance(v['tvec'], np.ndarray) else v['tvec']}
                               for k, v in self.camera_poses.items()},
                'frame_aruco_count': filtered_frame_aruco_count,  # Only include frames with camera poses
                'camera_intrinsic': self.camera_intrinsic.tolist() if isinstance(self.camera_intrinsic, np.ndarray) else self.camera_intrinsic,
                'image_name_to_frame_id': image_name_to_frame_id  # Mapping for pose remapping
            },
            'keypoints': output_data
        }
        
        # Save with error handling
        try:
            if USE_UTILS:
                safe_write_json(output_structure, output_path, indent=2)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_structure, f, indent=2)
        except (PermissionError, OSError) as e:
            QMessageBox.critical(self, "Error", f"Failed to save JSON file: {e}")
            raise
        
        return output_path
    
    def on_save_label_clicked(self):
        """Handle Save Label button click - calculate 3D/2D for each image with >2 ArUco markers"""
        # Allow saving even without keypoints - just save poses data
        if self.labeling_data or self.labeling_tab.calculated_2d:
            self.labeling_tab.log_message("Calculating 3D and 2D keypoint locations...")
        else:
            self.labeling_tab.log_message("No keypoints labeled - saving poses data only...")
        
        # Calculate 3D keypoints for each frame with >2 ArUco markers
        output_data = []
        
        # First, save user-labeled keypoints
        for frame_id in range(len(self.images)):
            # Check if frame has >2 ArUco markers
            aruco_count = self.frame_aruco_count.get(frame_id, 0)
            if aruco_count <= 2:
                continue

            # Check if frame has camera pose
            if frame_id not in self.camera_poses:
                continue

            # Process user-labeled keypoints in this frame
            if frame_id in self.labeling_data:
                camera_pose = self.camera_poses[frame_id]
                rvec = np.array(camera_pose['rvec'])
                tvec = np.array(camera_pose['tvec'])

                # Track which objects are visible (to avoid checking multiple times)
                visible_objects = {}  # {object_id: bool}

                for object_id, keypoints in self.labeling_data[frame_id].items():
                    # Check visibility for this object (only if checkbox is enabled)
                    if self.labeling_tab.check_visibility_checkbox.isChecked():
                        if object_id not in visible_objects:
                            visible_objects[object_id] = self.labeling_tab.check_object_visibility(frame_id, object_id)
                    else:
                        # If visibility check is disabled, consider all objects visible
                        visible_objects[object_id] = True
                    
                    for keypoint_id, data in keypoints.items():
                        if data.get('2d') is not None:
                            point_2d = np.array(data['2d'])

                            # Use triangulated 3D if available, otherwise calculate
                            if (object_id, keypoint_id) in self.labeling_tab.calculated_3d:
                                point_3d = np.array(self.labeling_tab.calculated_3d[(object_id, keypoint_id)])
                            else:
                                # Calculate 3D location using back-projection
                                point_3d = self.calculate_3d_from_2d(
                                    point_2d, rvec, tvec, self.camera_intrinsic
                                )

                            # Get visibility status: use stored value from JSON if present, otherwise calculate
                            if 'visibility' in data and data['visibility'] is not None:
                                is_visible = bool(data['visibility'])
                            else:
                                # Calculate visibility if not stored
                                is_visible = bool(visible_objects.get(object_id, True))

                            # Get image name from frame_id with validation
                            try:
                                image_name = get_image_name_safe(self.image_paths, frame_id)
                            except IndexError:
                                image_name = f"frame_{frame_id}.jpg"
                            output_data.append({
                                'image_name': image_name,
                                'object_id': object_id,
                                'keypoint_id': keypoint_id,
                                'keypoint_3d': point_3d.tolist(),
                                'keypoint_2d': point_2d.tolist(),
                                'is_manual': 1,  # User-provided keypoint
                                'cls_id': data.get('cls_id', 1),  # Default to 1 if not present (backward compatibility)
                                'visibility': is_visible  # Visibility status
                            })
        
        # Second, save calculated keypoints (from triangulation)
        for frame_id in range(len(self.images)):
            # Check if frame has >2 ArUco markers
            aruco_count = self.frame_aruco_count.get(frame_id, 0)
            if aruco_count <= 2:
                continue

            # Check if frame has camera pose
            if frame_id not in self.camera_poses:
                continue

            # Process calculated keypoints in this frame
            if frame_id in self.labeling_tab.calculated_2d:
                # Track which objects are visible (to avoid checking multiple times)
                visible_objects = {}  # {object_id: bool}
                
                for object_id, keypoints in self.labeling_tab.calculated_2d[frame_id].items():
                    # Check visibility for this object (only if checkbox is enabled)
                    if self.labeling_tab.check_visibility_checkbox.isChecked():
                        if object_id not in visible_objects:
                            visible_objects[object_id] = self.labeling_tab.check_object_visibility(frame_id, object_id)
                    else:
                        # If visibility check is disabled, consider all objects visible
                        visible_objects[object_id] = True
                    
                    for keypoint_id, point_2d in keypoints.items():
                        # Only save if not already saved as user-labeled
                        is_user_labeled = (frame_id in self.labeling_data and 
                                          object_id in self.labeling_data[frame_id] and
                                          keypoint_id in self.labeling_data[frame_id][object_id])
                        
                        if not is_user_labeled:
                            # Use triangulated 3D location
                            if (object_id, keypoint_id) in self.labeling_tab.calculated_3d:
                                point_3d = np.array(self.labeling_tab.calculated_3d[(object_id, keypoint_id)])
                                point_2d_array = np.array(point_2d)

                                # Get visibility status: use stored value from JSON if present, otherwise calculate
                                visibility_key = (frame_id, object_id, keypoint_id)
                                if (hasattr(self.labeling_tab, 'calculated_visibility') and 
                                    self.labeling_tab.calculated_visibility and 
                                    visibility_key in self.labeling_tab.calculated_visibility):
                                    is_visible = bool(self.labeling_tab.calculated_visibility[visibility_key])
                                else:
                                    # Calculate visibility if not stored
                                    is_visible = bool(visible_objects.get(object_id, True))

                                # Try to get cls_id from any user-labeled keypoint with same object_id and keypoint_id
                                cls_id = 1  # Default
                                for f_id, objs in self.labeling_data.items():
                                    if object_id in objs and keypoint_id in objs[object_id]:
                                        cls_id = objs[object_id][keypoint_id].get('cls_id', 1)
                                        break

                                # Get image name from frame_id with validation
                                try:
                                    image_name = get_image_name_safe(self.image_paths, frame_id)
                                except IndexError:
                                    image_name = f"frame_{frame_id}.jpg"
                                output_data.append({
                                    'image_name': image_name,
                                    'object_id': object_id,
                                    'keypoint_id': keypoint_id,
                                    'keypoint_3d': point_3d.tolist(),
                                    'keypoint_2d': point_2d_array.tolist(),
                                    'is_manual': 0,  # Triangulated keypoint
                                    'cls_id': cls_id,
                                    'visibility': is_visible  # Visibility status
                                })
        
        # Allow saving even when there are no keypoints - just save poses data
        # if not output_data:
        #     QMessageBox.warning(self, "Warning", "No keypoints to save! Make sure frames have >2 ArUco markers.")
        #     return
        
        # Identify valid frames (frames with >2 ArUco markers and camera poses)
        valid_frame_ids = set()
        for frame_id in range(len(self.images)):
            aruco_count = self.frame_aruco_count.get(frame_id, 0)
            if aruco_count > 2 and frame_id in self.camera_poses:
                valid_frame_ids.add(frame_id)
        
        if not valid_frame_ids:
            QMessageBox.warning(self, "Warning", "No valid frames found! Make sure frames have >2 ArUco markers.")
            return
        
        # Create valid images directory
        valid_dir = f"{self.image_dir}_valid"
        os.makedirs(valid_dir, exist_ok=True)
        self.labeling_tab.log_message(f"Created directory for valid images: {valid_dir}")
        
        # Copy valid images to the new directory
        copied_count = 0
        for frame_id in valid_frame_ids:
            if frame_id < len(self.image_paths):
                source_path = self.image_paths[frame_id]
                if os.path.exists(source_path):
                    # Get the filename from the source path
                    filename = os.path.basename(source_path)
                    dest_path = os.path.join(valid_dir, filename)
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
        
        self.labeling_tab.log_message(f"Copied {copied_count} valid images to {valid_dir}")
        
        # Save JSON file in the valid directory
        output_path = os.path.join(valid_dir, 'labeling_results.json')
        
        # Create image_name to frame_id mapping for pose remapping
        image_name_to_frame_id = {}
        for frame_id in range(len(self.image_paths)):
            try:
                image_name = get_image_name_safe(self.image_paths, frame_id)
                image_name_to_frame_id[image_name] = frame_id
            except IndexError:
                continue  # Skip invalid frame IDs
        
        # Filter frame_aruco_count to only include frames that have camera poses (i.e., not filtered out)
        filtered_frame_aruco_count = {
            str(k): v for k, v in self.frame_aruco_count.items() 
            if k in self.camera_poses
        }
        
        # Create output structure with metadata
        output_structure = {
            'metadata': {
                'aruco_poses': {str(k): {'rvec': v['rvec'].tolist() if isinstance(v['rvec'], np.ndarray) else v['rvec'],
                                        'tvec': v['tvec'].tolist() if isinstance(v['tvec'], np.ndarray) else v['tvec']}
                               for k, v in self.aruco_poses.items()},
                'camera_poses': {str(k): {'rvec': v['rvec'].tolist() if isinstance(v['rvec'], np.ndarray) else v['rvec'],
                                        'tvec': v['tvec'].tolist() if isinstance(v['tvec'], np.ndarray) else v['tvec']}
                               for k, v in self.camera_poses.items()},
                'frame_aruco_count': filtered_frame_aruco_count,  # Only include frames with camera poses
                'camera_intrinsic': self.camera_intrinsic.tolist() if isinstance(self.camera_intrinsic, np.ndarray) else self.camera_intrinsic,
                'image_name_to_frame_id': image_name_to_frame_id  # Mapping for pose remapping
            },
            'keypoints': output_data
        }
        
        try:
            if USE_UTILS:
                safe_write_json(output_structure, output_path, indent=2)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_structure, f, indent=2)
        except (PermissionError, OSError) as e:
            QMessageBox.critical(self, "Error", f"Failed to save JSON file: {e}")
            raise
        
        # Track saved path
        self.saved_json_path = output_path
        self.labeling_tab.set_saved_json_path(output_path)
        
        # Count manual and triangulated keypoints using is_manual field
        user_count = sum(1 for item in output_data if item.get('is_manual', 0) == 1)
        calc_count = sum(1 for item in output_data if item.get('is_manual', 0) == 0)

        if output_data:
            self.labeling_tab.log_message(f"Saved {len(output_data)} keypoint labels to: {output_path}")
            self.labeling_tab.log_message(f"  - {user_count} user-labeled, {calc_count} calculated from triangulation")
            QMessageBox.information(self, "Success",
                                   f"Labeling data saved to:\n{output_path}\n\n"
                                   f"{len(output_data)} keypoints saved ({user_count} user-labeled, {calc_count} calculated).\n"
                                   f"{copied_count} valid images copied to:\n{valid_dir}")
        else:
            self.labeling_tab.log_message(f"Saved poses data (no keypoints) to: {output_path}")
            QMessageBox.information(self, "Success",
                                   f"Poses data saved to:\n{output_path}\n\n"
                                   f"No keypoints labeled - only camera poses and ArUco marker poses saved.\n"
                                   f"{copied_count} valid images copied to:\n{valid_dir}")
    
    def normalize_coordinates(self, x: float, y: float, img_width: int, img_height: int):
        """Normalize coordinates to [0, 1] range"""
        if img_width <= 0 or img_height <= 0:
            raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")
        return x / img_width, y / img_height
    
    def calculate_bbox_from_keypoints(self, keypoints_2d):
        """Calculate bounding box from keypoint coordinates"""
        if not keypoints_2d:
            return None
        xs = [kp[0] for kp in keypoints_2d if len(kp) >= 2]
        ys = [kp[1] for kp in keypoints_2d if len(kp) >= 2]
        if not xs or not ys:
            return None
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            return None
        padding_x = width * 0.01
        padding_y = height * 0.01
        return (max(0, x_min - padding_x), max(0, y_min - padding_y), 
                x_max + padding_x, y_max + padding_y)
    
    def on_save_yolo_clicked(self):
        """Handle Save YOLO pose format button click"""
        # Check if we have valid data
        if not self.images or not self.image_paths:
            QMessageBox.warning(self, "Warning", "No images loaded!")
            return
        
        # Identify valid frames (frames with >2 ArUco markers and camera poses)
        valid_frame_ids = set()
        for frame_id in range(len(self.images)):
            aruco_count = self.frame_aruco_count.get(frame_id, 0)
            if aruco_count > 2 and frame_id in self.camera_poses:
                valid_frame_ids.add(frame_id)
        
        if not valid_frame_ids:
            QMessageBox.warning(self, "Warning", "No valid frames found! Make sure frames have >2 ArUco markers.")
            return
        
        # Get input directory name
        input_dir_path = Path(self.image_dir)
        input_dir_name = input_dir_path.name
        
        # Create output directory
        output_dir = str(input_dir_path.parent / f"{input_dir_name}_yolo")
        image_dir = os.path.join(output_dir, "image")
        label_dir = os.path.join(output_dir, "label")
        
        try:
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
        except (PermissionError, OSError) as e:
            QMessageBox.critical(self, "Error", f"Failed to create output directory: {e}")
            return
        
        self.labeling_tab.log_message(f"Converting to YOLO format...")
        self.labeling_tab.log_message(f"Output directory: {output_dir}")
        
        # Process valid frames
        processed_count = 0
        skipped_count = 0
        
        for frame_id in sorted(valid_frame_ids):
            if frame_id >= len(self.images) or frame_id >= len(self.image_paths):
                skipped_count += 1
                continue
            
            # Get image dimensions
            img = self.images[frame_id]
            img_height, img_width = img.shape[:2]
            
            # Collect keypoints for this frame grouped by object_id
            objects_keypoints = defaultdict(list)  # {object_id: [(keypoint_id, x, y, cls_id, visibility)]}
            
            # Collect user-labeled keypoints
            if frame_id in self.labeling_data:
                visible_objects = {}
                for object_id, keypoints in self.labeling_data[frame_id].items():
                    # Check visibility
                    if self.labeling_tab.check_visibility_checkbox.isChecked():
                        if object_id not in visible_objects:
                            visible_objects[object_id] = self.labeling_tab.check_object_visibility(frame_id, object_id)
                    else:
                        visible_objects[object_id] = True
                    
                    if not visible_objects[object_id]:
                        continue
                    
                    for keypoint_id, data in keypoints.items():
                        if data.get('2d') is not None:
                            x, y = data['2d']
                            cls_id = data.get('cls_id', 1)
                            visibility = data.get('visibility', True)
                            visibility_int = 1 if visibility else 0
                            objects_keypoints[object_id].append((keypoint_id, x, y, cls_id, visibility_int))
            
            # Collect calculated keypoints (triangulated)
            if frame_id in self.labeling_tab.calculated_2d:
                visible_objects = {}
                for object_id, keypoints in self.labeling_tab.calculated_2d[frame_id].items():
                    # Skip if already user-labeled
                    is_user_labeled = (frame_id in self.labeling_data and 
                                      object_id in self.labeling_data[frame_id])
                    
                    if is_user_labeled:
                        continue
                    
                    # Check visibility
                    if self.labeling_tab.check_visibility_checkbox.isChecked():
                        if object_id not in visible_objects:
                            visible_objects[object_id] = self.labeling_tab.check_object_visibility(frame_id, object_id)
                    else:
                        visible_objects[object_id] = True
                    
                    if not visible_objects[object_id]:
                        continue
                    
                    for keypoint_id, point_2d in keypoints.items():
                        # Skip if user-labeled
                        if (is_user_labeled and keypoint_id in self.labeling_data[frame_id][object_id]):
                            continue
                        
                        x, y = point_2d
                        # Get cls_id from any user-labeled keypoint with same object_id and keypoint_id
                        cls_id = 1  # Default
                        for f_id, objs in self.labeling_data.items():
                            if object_id in objs and keypoint_id in objs[object_id]:
                                cls_id = objs[object_id][keypoint_id].get('cls_id', 1)
                                break
                        
                        # Get visibility from calculated_visibility if available
                        visibility_key = (frame_id, object_id, keypoint_id)
                        if (hasattr(self.labeling_tab, 'calculated_visibility') and 
                            self.labeling_tab.calculated_visibility and 
                            visibility_key in self.labeling_tab.calculated_visibility):
                            visibility = self.labeling_tab.calculated_visibility[visibility_key]
                            visibility_int = 1 if visibility else 0
                        else:
                            visibility_int = 1  # Default to visible
                        
                        objects_keypoints[object_id].append((keypoint_id, x, y, cls_id, visibility_int))
            
            # Create YOLO format label file (create even if empty, but skip writing if no keypoints)
            label_lines = []
            
            for object_id, keypoints_list in objects_keypoints.items():
                # Convert keypoints_list to a dictionary for easier lookup
                # Tuple format: (keypoint_id, x, y, cls_id, visibility_int)
                keypoints_dict = {kp_id: (x, y, cls_id, visibility) for kp_id, x, y, cls_id, visibility in keypoints_list}
                
                if not keypoints_dict:
                    continue
                
                # Get class ID for this object
                # Use cls_id from first keypoint, but prefer user-labeled keypoints if available
                cls_id = next(iter(keypoints_dict.values()))[2]  # cls_id is at index 2 in tuple
                
                # If available, try to get cls_id from user-labeled keypoints for better accuracy
                if frame_id in self.labeling_data and object_id in self.labeling_data[frame_id]:
                    # Get cls_id from any user-labeled keypoint in this object
                    for kp_id, data in self.labeling_data[frame_id][object_id].items():
                        if data.get('2d') is not None and 'cls_id' in data:
                            cls_id = data.get('cls_id', cls_id)
                            break  # Use first found user-labeled keypoint's cls_id
                
                class_index = cls_id - 1  # YOLO uses 0-indexed classes
                
                # Extract 2D coordinates and visibility, filling missing keypoints with (0, 0, 0)
                # Keypoints must be in order from 1 to MAX_KEYPOINT_NUM
                keypoints_2d = []
                keypoints_visibility = []
                for kp_id in range(1, MAX_KEYPOINT_NUM + 1):
                    if kp_id in keypoints_dict:
                        x, y, _, visibility = keypoints_dict[kp_id]
                        keypoints_2d.append([float(x), float(y)])
                        keypoints_visibility.append(visibility)
                    else:
                        # Fill missing keypoint with (0, 0, 0)
                        keypoints_2d.append([0.0, 0.0])
                        keypoints_visibility.append(0)
                
                # Calculate bounding box (only from non-zero keypoints)
                non_zero_keypoints = [kp for kp in keypoints_2d if kp[0] != 0.0 or kp[1] != 0.0]
                if not non_zero_keypoints:
                    continue
                
                bbox = self.calculate_bbox_from_keypoints(non_zero_keypoints)
                if bbox is None:
                    continue
                
                x_min, y_min, x_max, y_max = bbox
                
                # Calculate normalized bbox center and size
                bbox_center_x = (x_min + x_max) / 2.0
                bbox_center_y = (y_min + y_max) / 2.0
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                
                # Normalize bbox coordinates
                try:
                    norm_center_x, norm_center_y = self.normalize_coordinates(
                        bbox_center_x, bbox_center_y, img_width, img_height
                    )
                    norm_width = bbox_width / img_width
                    norm_height = bbox_height / img_height
                except ValueError:
                    continue
                
                # Normalize keypoint coordinates (all keypoints from 1 to MAX_KEYPOINT_NUM)
                normalized_keypoints = []
                for idx, kp_2d in enumerate(keypoints_2d):
                    try:
                        # For missing keypoints (0, 0), keep them as (0, 0, 0)
                        if kp_2d[0] == 0.0 and kp_2d[1] == 0.0:
                            normalized_keypoints.append(0.0)
                            normalized_keypoints.append(0.0)
                            normalized_keypoints.append(0)
                        else:
                            norm_x, norm_y = self.normalize_coordinates(
                                kp_2d[0], kp_2d[1], img_width, img_height
                            )
                            # Clamp to [0, 1] range
                            norm_x = max(0.0, min(1.0, norm_x))
                            norm_y = max(0.0, min(1.0, norm_y))
                            normalized_keypoints.append(norm_x)
                            normalized_keypoints.append(norm_y)
                            visibility = keypoints_visibility[idx] if idx < len(keypoints_visibility) else 0
                            normalized_keypoints.append(visibility)
                    except (ValueError, TypeError, ZeroDivisionError):
                        # If normalization fails, fill with (0, 0, 0)
                        normalized_keypoints.append(0.0)
                        normalized_keypoints.append(0.0)
                        normalized_keypoints.append(0)
                
                # Format: <class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> ...
                line_parts = [
                    str(class_index),
                    f"{norm_center_x:.6f}",
                    f"{norm_center_y:.6f}",
                    f"{norm_width:.6f}",
                    f"{norm_height:.6f}"
                ]
                
                # Add keypoints (all keypoints from 1 to MAX_KEYPOINT_NUM in order)
                for i in range(0, len(normalized_keypoints), 3):
                    line_parts.append(f"{normalized_keypoints[i]:.6f}")
                    line_parts.append(f"{normalized_keypoints[i+1]:.6f}")
                    line_parts.append(f"{normalized_keypoints[i+2]:.0f}")
                
                label_lines.append(" ".join(line_parts))
            
            # Write label file only if there are keypoints (only valid labels)
            if label_lines:
                label_filename = f"{input_dir_name}_{frame_id}.txt"
                label_path = os.path.join(label_dir, label_filename)
                image_filename = f"{input_dir_name}_{frame_id}.jpg"
                image_path = os.path.join(image_dir, image_filename)
                try:
                    # Write label file
                    with open(label_path, 'w', encoding='utf-8') as f:
                        f.write("\n".join(label_lines))
                    
                    # Copy image (only if there are labels)
                    cv2.imwrite(image_path, img)
                    
                    processed_count += 1
                except (PermissionError, OSError) as e:
                    self.labeling_tab.log_message(f"Error writing files for frame {frame_id}: {e}")
                    skipped_count += 1
            else:
                skipped_count += 1
        
        self.labeling_tab.log_message(f"YOLO format conversion complete!")
        self.labeling_tab.log_message(f"  Processed: {processed_count} frames")
        self.labeling_tab.log_message(f"  Skipped: {skipped_count} frames")
        self.labeling_tab.log_message(f"  Output directory: {output_dir}")
        
        # Track saved YOLO path
        self.saved_yolo_path = output_dir
        self.labeling_tab.set_saved_yolo_path(output_dir)
        
        QMessageBox.information(self, "Success",
                               f"YOLO format saved to:\n{output_dir}\n\n"
                               f"Processed: {processed_count} frames\n"
                               f"Skipped: {skipped_count} frames")
    
    def on_exit_requested(self):
        """Handle exit request - check if data has been saved"""
        # Check if there's any labeling data or calculated data
        has_labeling_data = bool(self.labeling_data)
        has_calculated_data = bool(self.labeling_tab.calculated_2d or self.labeling_tab.calculated_3d)
        
        # Check if data has been saved (either JSON or YOLO format)
        json_saved = self.saved_json_path is not None or self.loaded_json_path is not None
        yolo_saved = self.saved_yolo_path is not None
        data_saved = json_saved or yolo_saved
        
        # If there's data but nothing saved, prompt user
        if (has_labeling_data or has_calculated_data) and not data_saved:
            reply = QMessageBox.warning(
                self,
                "Unsaved Changes",
                "You have unsaved labeling data. Please save your work before exiting.\n\n"
                "Click 'Yes' to exit without saving, or 'No' to cancel and save your work.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return  # User cancelled
        
        # Exit the application
        self.close()
    
    def calculate_3d_from_2d(self, point_2d, rvec, tvec, camera_intrinsic):
        """
        Calculate 3D point from 2D using back-projection with estimated depth
        Depth is estimated from average distance to ArUco markers
        """
        # Convert 2D to normalized coordinates
        fx = camera_intrinsic[0, 0]
        fy = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        
        x_norm = (point_2d[0] - cx) / fx
        y_norm = (point_2d[1] - cy) / fy
        
        # Estimate depth from ArUco markers
        # Use average distance from camera to ArUco markers as depth estimate
        R, _ = cv2.Rodrigues(rvec)
        depth = 1.0  # Default depth
        
        if self.aruco_poses:
            # Calculate average distance to ArUco markers
            distances = []
            for marker_id, marker_pose in self.aruco_poses.items():
                marker_tvec = np.array(marker_pose['tvec'])
                # Distance from camera to marker in world frame
                marker_pos_world = marker_tvec
                camera_pos_world = -R.T @ tvec
                dist = np.linalg.norm(marker_pos_world - camera_pos_world)
                distances.append(dist)
            
            if distances:
                depth = np.mean(distances)
        
        # Back-project to 3D in camera frame
        point_3d_cam = np.array([x_norm * depth, y_norm * depth, depth])
        
        # Transform to world coordinates
        # Camera to world: R_world_cam = R^T, t_world_cam = -R^T @ t
        point_3d_world = R.T @ point_3d_cam - R.T @ tvec
        
        return point_3d_world

