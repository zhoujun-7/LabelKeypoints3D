"""
Joint optimization of 3D and 2D keypoint locations
"""

import numpy as np
from scipy.optimize import least_squares
import cv2
from typing import Dict, Any, List, Tuple

# Try to import shared triangulation function
try:
    from .utils import triangulate_points
    USE_SHARED_TRIANGULATION = True
except ImportError:
    USE_SHARED_TRIANGULATION = False


class KeypointOptimizer:
    def __init__(self, labeling_data, camera_poses, camera_intrinsic):
        """
        Initialize optimizer
        
        Args:
            labeling_data: {frame_id: {object_id: {keypoint_id: {'2d': [x, y], '3d': [x, y, z]}}}}
            camera_poses: {frame_id: {'rvec': [...], 'tvec': [...]}} - per-frame camera poses
            camera_intrinsic: 3x3 camera intrinsic matrix
        """
        self.labeling_data = labeling_data
        self.camera_poses = camera_poses
        self.camera_intrinsic = camera_intrinsic
        
        # Extract all unique (object_id, keypoint_id) pairs
        self.keypoint_pairs = set()
        for frame_id, objects in labeling_data.items():
            for object_id, keypoints in objects.items():
                for keypoint_id in keypoints.keys():
                    self.keypoint_pairs.add((object_id, keypoint_id))
        
        self.keypoint_pairs = sorted(list(self.keypoint_pairs))
    
    def optimize(self):
        """
        Jointly optimize 3D keypoint locations
        
        Returns:
            Updated labeling_data with optimized 3D locations
        """
        if not self.keypoint_pairs:
            return self.labeling_data
        
        # Initialize 3D locations (triangulate from first two observations)
        keypoint_3d_locations = {}
        for obj_id, kp_id in self.keypoint_pairs:
            # Find first two frames with this keypoint
            frames_with_kp = []
            for frame_id, objects in self.labeling_data.items():
                if (obj_id in objects and 
                    kp_id in objects[obj_id] and 
                    objects[obj_id][kp_id]['2d'] is not None):
                    frames_with_kp.append(frame_id)
            
            if len(frames_with_kp) >= 2:
                # Triangulate from first two frames
                frame1, frame2 = frames_with_kp[0], frames_with_kp[1]
                pt1_2d = np.array(self.labeling_data[frame1][obj_id][kp_id]['2d'])
                pt2_2d = np.array(self.labeling_data[frame2][obj_id][kp_id]['2d'])
                
                # Get camera poses for both frames
                if frame1 in self.camera_poses and frame2 in self.camera_poses:
                    rvec1 = np.array(self.camera_poses[frame1]['rvec'])
                    tvec1 = np.array(self.camera_poses[frame1]['tvec'])
                    rvec2 = np.array(self.camera_poses[frame2]['rvec'])
                    tvec2 = np.array(self.camera_poses[frame2]['tvec'])
                    
                    R1, _ = cv2.Rodrigues(rvec1)
                    R2, _ = cv2.Rodrigues(rvec2)
                    
                    # Triangulate using shared utility function
                    P1 = self.camera_intrinsic @ np.hstack([R1, tvec1.reshape(3, 1)])
                    P2 = self.camera_intrinsic @ np.hstack([R2, tvec2.reshape(3, 1)])
                    
                    try:
                        if USE_SHARED_TRIANGULATION:
                            pt_3d = triangulate_points(pt1_2d, pt2_2d, P1, P2)
                        else:
                            pt_3d = self.triangulate_points(pt1_2d, pt2_2d, P1, P2)
                        keypoint_3d_locations[(obj_id, kp_id)] = pt_3d
                    except ValueError:
                        # Triangulation failed, use fallback
                        keypoint_3d_locations[(obj_id, kp_id)] = np.array([0.0, 0.0, 1.0])
                else:
                    # Fallback: initialize at origin
                    keypoint_3d_locations[(obj_id, kp_id)] = np.array([0.0, 0.0, 1.0])
            else:
                # Initialize at origin
                keypoint_3d_locations[(obj_id, kp_id)] = np.array([0.0, 0.0, 1.0])
        
        # Prepare optimization parameters
        # Parameters: [x1, y1, z1, x2, y2, z2, ...] for each keypoint
        params = []
        for obj_id, kp_id in self.keypoint_pairs:
            params.extend(keypoint_3d_locations[(obj_id, kp_id)])
        params = np.array(params)
        
        # Prepare observations
        observations = []  # List of (frame_id, obj_id, kp_id, point_2d)
        for frame_id, objects in self.labeling_data.items():
            for obj_id, keypoints in objects.items():
                for kp_id, data in keypoints.items():
                    if data['2d'] is not None:
                        observations.append((frame_id, obj_id, kp_id, np.array(data['2d'])))
        
        if not observations:
            return self.labeling_data
        
        # Optimization function
        def residuals(params):
            # Extract 3D locations
            keypoint_3d = {}
            for idx, (obj_id, kp_id) in enumerate(self.keypoint_pairs):
                keypoint_3d[(obj_id, kp_id)] = params[idx*3:(idx+1)*3]
            
            residuals_list = []
            
            for frame_id, obj_id, kp_id, point_2d in observations:
                if (obj_id, kp_id) not in keypoint_3d:
                    continue
                
                pt_3d = keypoint_3d[(obj_id, kp_id)]
                
                # Get camera pose for this frame
                if frame_id in self.camera_poses:
                    rvec = np.array(self.camera_poses[frame_id]['rvec'])
                    tvec = np.array(self.camera_poses[frame_id]['tvec'])
                else:
                    # Fallback: use identity
                    rvec = np.zeros(3)
                    tvec = np.zeros(3)
                
                # Project 3D point to 2D
                projected, _ = cv2.projectPoints(
                    pt_3d.reshape(1, 3),
                    rvec,
                    tvec,
                    self.camera_intrinsic,
                    None
                )
                projected_2d = projected[0, 0]
                
                # Compute residual
                residual = point_2d - projected_2d
                residuals_list.extend(residual)
            
            return np.array(residuals_list)
        
        # Optimize
        result = least_squares(residuals, params, method='lm', verbose=0, max_nfev=1000)
        
        # Update labeling data with optimized 3D locations
        optimized_data = {}
        for frame_id, objects in self.labeling_data.items():
            optimized_data[frame_id] = {}
            for obj_id, keypoints in objects.items():
                optimized_data[frame_id][obj_id] = {}
                for kp_id, data in keypoints.items():
                    optimized_data[frame_id][obj_id][kp_id] = data.copy()
                    
                    # Update 3D location
                    if (obj_id, kp_id) in self.keypoint_pairs:
                        idx = self.keypoint_pairs.index((obj_id, kp_id))
                        pt_3d = result.x[idx*3:(idx+1)*3]
                        optimized_data[frame_id][obj_id][kp_id]['3d'] = pt_3d.tolist()
        
        return optimized_data
    
    @staticmethod
    def triangulate_points(pt1: np.ndarray, pt2: np.ndarray, 
                          P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D point from two 2D observations (fallback if utils not available)
        
        Args:
            pt1: 2D point in first image [x, y]
            pt2: 2D point in second image [x, y]
            P1: 3x4 projection matrix for first camera
            P2: 3x4 projection matrix for second camera
        
        Returns:
            3D point [x, y, z]
        
        Raises:
            ValueError: If triangulation fails
        """
        # Build system of equations
        A = np.zeros((4, 4))
        A[0] = pt1[0] * P1[2] - P1[0]
        A[1] = pt1[1] * P1[2] - P1[1]
        A[2] = pt2[0] * P2[2] - P2[0]
        A[3] = pt2[1] * P2[2] - P2[1]
        
        # Solve using SVD
        try:
            _, _, V = np.linalg.svd(A)
            pt_3d = V[-1]
            pt_3d = pt_3d[:3] / pt_3d[3]
            return pt_3d
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Triangulation failed: {e}")

