"""
ArUco marker detection and pose estimation with bundle adjustment
"""

from typing import Any
from PySide6.QtCore import QThread, Signal
import cv2
import numpy as np
import os
import sys
from scipy.optimize import least_squares
import glob
import time
import torch
import multiprocessing as mp
from .bundle_adjustment import BundleAdjustment

# Import default configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import default_cfg
    DEFAULT_FX = default_cfg.fx
    DEFAULT_FY = default_cfg.fy
    DEFAULT_CX = default_cfg.cx
    DEFAULT_CY = default_cfg.cy
    MIN_ARUCO_COUNT = default_cfg.min_aruco_count
    ISOLATION_NEIGHBOR_NUM = default_cfg.isolation_neighboor_num
    # ArUco detector parameters
    ARUCO_MIN_CORNER_DISTANCE_RATE = default_cfg.aruco_min_corner_distance_rate
    ARUCO_MIN_MARKER_DISTANCE_RATE = default_cfg.aruco_min_marker_distance_rate
    ARUCO_POLYGONAL_APPROX_ACCURACY_RATE = default_cfg.aruco_polygonal_approx_accuracy_rate
    ARUCO_MIN_MARKER_PERIMETER_RATE = default_cfg.aruco_min_marker_perimeter_rate
    ARUCO_MAX_MARKER_PERIMETER_RATE = default_cfg.aruco_max_marker_perimeter_rate
    ARUCO_MIN_OTSU_STD_DEV = default_cfg.aruco_min_otsu_std_dev
    ARUCO_ADAPTIVE_THRESH_CONSTANT = default_cfg.aruco_adaptive_thresh_constant
    ARUCO_ERROR_CORRECTION_RATE = default_cfg.aruco_error_correction_rate
    BA_LEARNING_RATE = default_cfg.ba_learning_rate
    BA_MAX_ITERATIONS = default_cfg.ba_max_iterations
    REPROJECTION_ERROR_THRESHOLD = default_cfg.reprojection_error_threshold
except ImportError:
    # Fallback defaults if config file not found
    DEFAULT_FX = 1000.0
    DEFAULT_FY = 1000.0
    DEFAULT_CX = 540.0
    DEFAULT_CY = 360.0
    MIN_ARUCO_COUNT = 3
    ISOLATION_NEIGHBOR_NUM = 2
    # ArUco detector parameters (fallback)
    ARUCO_MIN_CORNER_DISTANCE_RATE = 0.1
    ARUCO_MIN_MARKER_DISTANCE_RATE = 0.1
    ARUCO_POLYGONAL_APPROX_ACCURACY_RATE = 0.02
    ARUCO_MIN_MARKER_PERIMETER_RATE = 0.05
    ARUCO_MAX_MARKER_PERIMETER_RATE = 2.0
    ARUCO_MIN_OTSU_STD_DEV = 7.0
    ARUCO_ADAPTIVE_THRESH_CONSTANT = 5
    ARUCO_ERROR_CORRECTION_RATE = 0.3
    BA_LEARNING_RATE = 1e-3
    BA_MAX_ITERATIONS = 1000
    REPROJECTION_ERROR_THRESHOLD = 10.0


def _detect_aruco_worker(args):
    """
    Worker function for multiprocessing ArUco detection.
    
    Args:
        args: tuple of (img_idx, img, aruco_params, camera_intrinsic, marker_size)
    
    Returns:
        tuple: (img_idx, detections_list, detection_count)
    """
    img_idx, img, aruco_params, camera_intrinsic, marker_size = args
    
    # Recreate detector in worker process (cannot share OpenCV objects across processes)
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, aruco_params['dict_type'])
    )
    parameters = cv2.aruco.DetectorParameters()
    
    # Configure parameters to suppress false positives
    parameters.minCornerDistanceRate = ARUCO_MIN_CORNER_DISTANCE_RATE
    parameters.minMarkerDistanceRate = ARUCO_MIN_MARKER_DISTANCE_RATE
    parameters.polygonalApproxAccuracyRate = ARUCO_POLYGONAL_APPROX_ACCURACY_RATE
    parameters.minMarkerPerimeterRate = ARUCO_MIN_MARKER_PERIMETER_RATE
    parameters.maxMarkerPerimeterRate = ARUCO_MAX_MARKER_PERIMETER_RATE
    parameters.minOtsuStdDev = ARUCO_MIN_OTSU_STD_DEV
    parameters.adaptiveThreshConstant = ARUCO_ADAPTIVE_THRESH_CONSTANT
    parameters.errorCorrectionRate = ARUCO_ERROR_CORRECTION_RATE
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Detect markers
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    detections = []
    detection_count = 0
    
    if ids is not None:
        detection_count = len(ids.flatten())
        # Estimate pose for each marker
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i][0]
            
            # Generate 3D object points
            obj_points = np.array([
                [-marker_size/2, marker_size/2, 0],
                [marker_size/2, marker_size/2, 0],
                [marker_size/2, -marker_size/2, 0],
                [-marker_size/2, -marker_size/2, 0]
            ], dtype=np.float32)
            
            # Estimate pose
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                marker_corners,
                camera_intrinsic,
                None
            )
            
            if success:
                detections.append({
                    'image_idx': img_idx,
                    'marker_id': int(marker_id),
                    'corners': marker_corners,
                    'rvec': rvec.flatten(),
                    'tvec': tvec.flatten()
                })
    
    return (img_idx, detections, detection_count)


class ArucoProcessor(QThread):
    progress_signal = Signal(str, int, int)  # message, current, total (0, 0 for indeterminate)
    finished_signal = Signal(dict)
    
    def __init__(self, image_dir, aruco_params, camera_intrinsic, enable_ba=True):
        super().__init__()
        self.image_dir = image_dir
        self.aruco_params = aruco_params
        self.enable_ba = enable_ba
        # Use default camera intrinsic if not provided
        if camera_intrinsic is None:
            # Default from config file
            self.camera_intrinsic = np.array([
                [DEFAULT_FX, 0, DEFAULT_CX],
                [0, DEFAULT_FY, DEFAULT_CY],
                [0, 0, 1]
            ], dtype=np.float64)
            print(f"[ARUCO PROCESSOR] Using default camera intrinsic from config: fx={DEFAULT_FX}, fy={DEFAULT_FY}, cx={DEFAULT_CX}, cy={DEFAULT_CY}")
        else:
            self.camera_intrinsic = camera_intrinsic
    
    def run(self):
        try:
            print(f"\n[ARUCO PROCESSOR] Starting ArUco marker detection...")
            print(f"[ARUCO PROCESSOR] Image directory: {self.image_dir}")
            print(f"[ARUCO PROCESSOR] ArUco parameters: {self.aruco_params}")
            print(f"[ARUCO PROCESSOR] Camera intrinsic provided: {self.camera_intrinsic is not None}")
            
            # Load images
            self.progress_signal.emit("Loading images...", 0, 0)
            image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")) +
                                glob.glob(os.path.join(self.image_dir, "*.png")) +
                                glob.glob(os.path.join(self.image_dir, "*.jpeg")))
            
            if not image_paths:
                print(f"[ARUCO PROCESSOR] ERROR: No images found in directory")
                self.finished_signal.emit({'success': False, 'error': 'No images found in directory'})
                return
            
            print(f"[ARUCO PROCESSOR] Found {len(image_paths)} image files")
            
            images = []
            total_images = len(image_paths)
            for idx, path in enumerate(image_paths):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                # Update progress every 10 images or at the end
                if (idx + 1) % 10 == 0 or (idx + 1) == total_images:
                    self.progress_signal.emit(f"Loading images... ({idx + 1}/{total_images})", idx + 1, total_images)
            
            if not images:
                print(f"[ARUCO PROCESSOR] ERROR: Failed to load images")
                self.finished_signal.emit({'success': False, 'error': 'Failed to load images'})
                return
            
            print(f"[ARUCO PROCESSOR] Successfully loaded {len(images)} images")
            self.progress_signal.emit(f"Loaded {len(images)} images", len(images), len(images))
            
            # Initialize ArUco detector with improved parameters to suppress false positives
            aruco_dict = cv2.aruco.getPredefinedDictionary(
                getattr(cv2.aruco, self.aruco_params['dict_type'])
            )
            parameters = cv2.aruco.DetectorParameters()
            
            # Configure parameters to suppress false positives
            parameters.minCornerDistanceRate = ARUCO_MIN_CORNER_DISTANCE_RATE
            parameters.minMarkerDistanceRate = ARUCO_MIN_MARKER_DISTANCE_RATE
            parameters.polygonalApproxAccuracyRate = ARUCO_POLYGONAL_APPROX_ACCURACY_RATE
            parameters.minMarkerPerimeterRate = ARUCO_MIN_MARKER_PERIMETER_RATE
            parameters.maxMarkerPerimeterRate = ARUCO_MAX_MARKER_PERIMETER_RATE
            parameters.minOtsuStdDev = ARUCO_MIN_OTSU_STD_DEV
            parameters.adaptiveThreshConstant = ARUCO_ADAPTIVE_THRESH_CONSTANT
            parameters.errorCorrectionRate = ARUCO_ERROR_CORRECTION_RATE
            
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            print(f"[ARUCO PROCESSOR] Using improved detector parameters to suppress false positives")
            print(f"[ARUCO PROCESSOR]   minCornerDistanceRate: {ARUCO_MIN_CORNER_DISTANCE_RATE}")
            print(f"[ARUCO PROCESSOR]   minMarkerDistanceRate: {ARUCO_MIN_MARKER_DISTANCE_RATE}")
            print(f"[ARUCO PROCESSOR]   errorCorrectionRate: {ARUCO_ERROR_CORRECTION_RATE}")
            
            # Detect ArUco markers in all images using multiprocessing
            self.progress_signal.emit("Detecting ArUco markers...", 0, len(images))
            print(f"[ARUCO PROCESSOR] Detecting ArUco markers in {len(images)} images...")
            print(f"[ARUCO PROCESSOR] Using provided camera intrinsic:")
            print(f"[ARUCO PROCESSOR]   fx={self.camera_intrinsic[0,0]:.2f}, fy={self.camera_intrinsic[1,1]:.2f}")
            print(f"[ARUCO PROCESSOR]   cx={self.camera_intrinsic[0,2]:.2f}, cy={self.camera_intrinsic[1,2]:.2f}")
            all_detections = []  # List of (image_idx, marker_id, corners, rvec, tvec)
            marker_size = self.aruco_params['physical_length']
            
            # Determine number of processes to use
            num_processes = min(mp.cpu_count(), len(images), 8)  # Limit to 8 processes max
            print(f"[ARUCO PROCESSOR] Using {num_processes} processes for parallel detection")
            
            # Prepare arguments for worker function
            worker_args = [
                (img_idx, img, self.aruco_params, self.camera_intrinsic, marker_size)
                for img_idx, img in enumerate(images)
            ]
            
            frame_detection_counts = {}
            total_images = len(images)
            
            # Use multiprocessing Pool for parallel detection
            if num_processes > 1 and len(images) > 1:
                try:
                    # Use multiprocessing for multiple images
                    with mp.Pool(processes=num_processes) as pool:
                        # Process images in batches to update progress
                        batch_size = max(1, total_images // 20)  # Update progress ~20 times
                        results = []
                        
                        # Use imap_unordered for better progress tracking
                        for idx, result in enumerate(pool.imap_unordered(_detect_aruco_worker, worker_args)):
                            results.append(result)
                            
                            # Update progress
                            if (idx + 1) % batch_size == 0 or (idx + 1) == total_images:
                                self.progress_signal.emit(
                                    f"Detecting ArUco markers... ({idx + 1}/{total_images})", 
                                    idx + 1, 
                                    total_images
                                )
                        
                        # Sort results by image index to maintain order
                        results.sort(key=lambda x: x[0])
                        
                        # Collect detections and counts
                        for img_idx, detections, detection_count in results:
                            frame_detection_counts[img_idx] = detection_count
                            all_detections.extend(detections)
                    
                    print(f"[ARUCO PROCESSOR] Parallel detection completed successfully")
                except Exception as e:
                    # Fallback to sequential processing if multiprocessing fails
                    print(f"[ARUCO PROCESSOR] WARNING: Multiprocessing failed ({str(e)}), falling back to sequential processing")
                    num_processes = 1  # Force sequential processing
            if num_processes == 1 or len(images) == 1:
                # Fallback to sequential processing for single image or single process
                print(f"[ARUCO PROCESSOR] Using sequential processing (num_processes={num_processes}, num_images={len(images)})")
                for img_idx, img in enumerate(images):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = detector.detectMarkers(gray)
                    
                    if ids is not None:
                        frame_detection_counts[img_idx] = len(ids.flatten())
                        # Estimate pose for each marker
                        for i, marker_id in enumerate(ids.flatten()):
                            marker_corners = corners[i][0]
                            
                            # Generate 3D object points
                            obj_points = np.array([
                                [-marker_size/2, marker_size/2, 0],
                                [marker_size/2, marker_size/2, 0],
                                [marker_size/2, -marker_size/2, 0],
                                [-marker_size/2, -marker_size/2, 0]
                            ], dtype=np.float32)
                            
                            # Estimate pose
                            success, rvec, tvec = cv2.solvePnP(
                                obj_points,
                                marker_corners,
                                self.camera_intrinsic,
                                None
                            )
                            
                            if success:
                                all_detections.append({
                                    'image_idx': img_idx,
                                    'marker_id': int(marker_id),
                                    'corners': marker_corners,
                                    'rvec': rvec.flatten(),
                                    'tvec': tvec.flatten()
                                })
                    else:
                        frame_detection_counts[img_idx] = 0
                    
                    # Update progress every 10 images or at the end
                    if (img_idx + 1) % 10 == 0 or (img_idx + 1) == total_images:
                        self.progress_signal.emit(f"Detecting ArUco markers... ({img_idx + 1}/{total_images})", img_idx + 1, total_images)
            
            if not all_detections:
                print(f"[ARUCO PROCESSOR] ERROR: No ArUco markers detected in any image")
                self.finished_signal.emit({'success': False, 'error': 'No ArUco markers detected'})
                return
            
            print(f"[ARUCO PROCESSOR] Detected {len(all_detections)} marker instances across {len(frame_detection_counts)} frames")
            print(f"[ARUCO PROCESSOR] Frames with markers: {sum(1 for c in frame_detection_counts.values() if c > 0)}")
            print(f"[ARUCO PROCESSOR] Frames without markers: {sum(1 for c in frame_detection_counts.values() if c == 0)}")
            self.progress_signal.emit(f"Detected {len(all_detections)} marker instances", total_images, total_images)
            
            # Filter images based on ArUco count threshold
            original_image_count = len(images)
            print(f"[ARUCO PROCESSOR] Filtering images with ArUco count >= {MIN_ARUCO_COUNT}...")
            valid_frame_indices = []
            old_to_new_idx = {}  # Mapping from old image index to new image index
            new_idx = 0
            
            for old_idx in range(len(images)):
                aruco_count = frame_detection_counts.get(old_idx, 0)
                if aruco_count >= MIN_ARUCO_COUNT:
                    valid_frame_indices.append(old_idx)
                    old_to_new_idx[old_idx] = new_idx
                    new_idx += 1
                else:
                    print(f"[ARUCO PROCESSOR] Ignoring image {old_idx} (ArUco count: {aruco_count} < {MIN_ARUCO_COUNT})")
            
            if not valid_frame_indices:
                print(f"[ARUCO PROCESSOR] ERROR: No images with at least {MIN_ARUCO_COUNT} ArUco markers")
                self.finished_signal.emit({'success': False, 'error': f'No images with at least {MIN_ARUCO_COUNT} ArUco markers'})
                return
            
            # Filter images and image_paths
            filtered_images = [images[i] for i in valid_frame_indices]
            filtered_image_paths = [image_paths[i] for i in valid_frame_indices]
            
            # Update detections: filter and remap image indices
            original_detection_count = len(all_detections)
            filtered_detections = []
            for det in all_detections:
                old_img_idx = det['image_idx']
                if old_img_idx in old_to_new_idx:
                    new_det = det.copy()
                    new_det['image_idx'] = old_to_new_idx[old_img_idx]
                    filtered_detections.append(new_det)
            
            ignored_count = original_image_count - len(filtered_images)
            print(f"[ARUCO PROCESSOR] Filtered: {original_image_count} -> {len(filtered_images)} images ({ignored_count} images ignored)")
            print(f"[ARUCO PROCESSOR] Filtered: {original_detection_count} -> {len(filtered_detections)} detections")
            self.progress_signal.emit(f"Filtered to {len(filtered_images)} images (ignored {ignored_count} images with <{MIN_ARUCO_COUNT} ArUco markers)", 0, 0)
            
            # Update images and image_paths for bundle adjustment
            images = filtered_images
            image_paths = filtered_image_paths
            all_detections = filtered_detections
            
            # Bundle adjustment (optional)
            ba = BundleAdjustment(
                self.camera_intrinsic,
                progress_callback=self.progress_signal.emit
            )
            
            if self.enable_ba:
                self.progress_signal.emit("Performing bundle adjustment...", 0, 0)
                print(f"[ARUCO PROCESSOR] Starting bundle adjustment with {len(all_detections)} detections...")
                optimized_result = ba.bundle_adjustment(all_detections, images, marker_size, enable_ba=True)
                print(f"[ARUCO PROCESSOR] Bundle adjustment completed")
                # Bundle adjustment may filter images, so use filtered images from result
                result_images = optimized_result.get('images', images)
            else:
                self.progress_signal.emit("Skipping bundle adjustment (disabled)...", 0, 0)
                print(f"[ARUCO PROCESSOR] Bundle adjustment disabled, using initial poses...")
                optimized_result = ba.bundle_adjustment(all_detections, images, marker_size, enable_ba=False)
                print(f"[ARUCO PROCESSOR] Using initial poses (no optimization)")
                # Bundle adjustment may filter images, so use filtered images from result
                result_images = optimized_result.get('images', images)
            
            # Extract ArUco poses (marker poses in world frame - stationary)
            # Get optimized marker world poses directly from bundle adjustment
            optimized_camera_poses = optimized_result.get('camera_poses', {})
            marker_poses_world = optimized_result.get('marker_poses_world', {})
            
            # Convert to list format for output
            aruco_poses = {}
            for marker_id, pose in marker_poses_world.items():
                aruco_poses[marker_id] = {
                    'rvec': pose['rvec'].tolist() if isinstance(pose['rvec'], np.ndarray) else pose['rvec'],
                    'tvec': pose['tvec'].tolist() if isinstance(pose['tvec'], np.ndarray) else pose['tvec']
                }
            
            # Count ArUco markers per frame
            frame_aruco_count = {}  # {frame_id: count}
            for img_idx in range(len(result_images)):
                frame_detections = [d for d in optimized_result['detections'] if d['image_idx'] == img_idx]
                frame_aruco_count[img_idx] = len(set(d['marker_id'] for d in frame_detections))
            
            # Use camera poses from bundle adjustment, fill in missing frames
            camera_poses = {}  # {frame_id: {'rvec': [...], 'tvec': [...]}}
            
            for img_idx in range(len(result_images)):
                if img_idx in optimized_camera_poses:
                    # Use optimized camera pose
                    cam_pose = optimized_camera_poses[img_idx]
                    camera_poses[img_idx] = {
                        'rvec': cam_pose['rvec'].tolist() if isinstance(cam_pose['rvec'], np.ndarray) else cam_pose['rvec'],
                        'tvec': cam_pose['tvec'].tolist() if isinstance(cam_pose['tvec'], np.ndarray) else cam_pose['tvec']
                    }
                else:
                    # No markers in this frame, use previous frame's pose or identity
                    if img_idx > 0 and (img_idx - 1) in camera_poses:
                        camera_poses[img_idx] = camera_poses[img_idx - 1].copy()
                    else:
                        camera_poses[img_idx] = {
                            'rvec': [0.0, 0.0, 0.0],
                            'tvec': [0.0, 0.0, 0.0]
                        }
            
            # Organize detections by frame for easy access
            frame_detections = {}  # {frame_id: [detection1, detection2, ...]}
            for det in optimized_result['detections']:
                frame_idx = det['image_idx']
                if frame_idx not in frame_detections:
                    frame_detections[frame_idx] = []
                frame_detections[frame_idx].append({
                    'marker_id': det['marker_id'],
                    'corners': det['corners'],  # 2D corners in image
                    'rvec': det['rvec'],  # Marker pose in camera frame
                    'tvec': det['tvec']
                })
            
            # Update image_paths to match filtered images if needed
            if not self.enable_ba and 'kept_frame_indices' in optimized_result:
                # Images were filtered, use kept_frame_indices to filter image_paths
                kept_indices = optimized_result['kept_frame_indices']
                filtered_image_paths = [image_paths[i] for i in kept_indices]
            else:
                filtered_image_paths = image_paths
            
            self.finished_signal.emit({
                'success': True,
                'aruco_poses': aruco_poses,
                'camera_poses': camera_poses,  # Per-frame camera poses
                'frame_aruco_count': frame_aruco_count,  # ArUco count per frame
                'frame_detections': frame_detections,  # Detections per frame
                'images': result_images,  # Use filtered images
                'image_paths': filtered_image_paths,  # Use filtered image paths
                'camera_intrinsic': optimized_result['camera_intrinsic'],
                'marker_size': marker_size  # Pass marker size for visualization
            })
            
        except Exception as e:
            self.finished_signal.emit({'success': False, 'error': str(e)})
