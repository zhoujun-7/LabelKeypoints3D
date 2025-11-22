"""
Bundle adjustment for ArUco marker pose estimation
"""

import cv2
import numpy as np
import time
import torch
import os
import sys

# Import default configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import default_cfg
    MIN_ARUCO_COUNT = default_cfg.min_aruco_count
    ISOLATION_NEIGHBOR_NUM = default_cfg.isolation_neighboor_num
    BA_LEARNING_RATE = default_cfg.ba_learning_rate
    BA_MAX_ITERATIONS = default_cfg.ba_max_iterations
    BA_DECAY_RATE = default_cfg.ba_decay_rate
    REPROJECTION_ERROR_THRESHOLD = default_cfg.reprojection_error_threshold
except ImportError:
    # Fallback defaults if config file not found
    MIN_ARUCO_COUNT = 3
    ISOLATION_NEIGHBOR_NUM = 2
    BA_LEARNING_RATE = 1e-3
    BA_MAX_ITERATIONS = 1000
    BA_DECAY_RATE = 0.998
    REPROJECTION_ERROR_THRESHOLD = 10.0


class BundleAdjustment:
    """
    Bundle adjustment for optimizing ArUco marker poses and camera poses.
    """
    
    def __init__(self, camera_intrinsic, progress_callback=None):
        """
        Initialize bundle adjustment.
        
        Args:
            camera_intrinsic: Camera intrinsic matrix (3x3 numpy array)
            progress_callback: Optional callback function for progress updates.
                              Signature: callback(message, current, total)
        """
        self.camera_intrinsic = camera_intrinsic
        self.progress_callback = progress_callback
    
    def _remove_outliers_iqr(self, values, axis=0):
        """
        Remove outliers using IQR (Interquartile Range) method.
        Returns mask of inlier indices.
        
        Args:
            values: Array of values (either translation vectors or rotation matrices)
            axis: 0 for translation vectors (shape: Nx3), 1 for rotation matrices (shape: Nx3x3)
        """
        if len(values) < 4:  # Need at least 4 values for IQR
            return np.ones(len(values), dtype=bool)
        
        # Compute distances from mean for each pose
        if axis == 0:
            # For translation vectors: compute L2 norm of differences from mean
            mean_val = np.mean(values, axis=0)
            distances = np.linalg.norm(values - mean_val, axis=1)
        else:
            # For rotation matrices: compute Frobenius norm of differences from mean
            # First compute mean rotation matrix
            mean_val = np.mean(values, axis=0)
            # Compute Frobenius norm of difference for each rotation matrix
            distances = np.array([np.linalg.norm(v - mean_val, 'fro') for v in values])
        
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        mask = (distances >= lower_bound) & (distances <= upper_bound)
        return mask

    def _compute_reprojection_error(self, obj_points_3d, image_points_2d, rvec, tvec, camera_intrinsic):
        """
        Compute mean reprojection error in pixels.
        """
        projected, _ = cv2.projectPoints(
            obj_points_3d,
            rvec,
            tvec,
            camera_intrinsic,
            None
        )
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(image_points_2d - projected, axis=1)
        return np.mean(errors)

    def _filter_false_positive_detections(self, detections, neighbor_num):
        """
        Filter false positive ArUco marker detections by checking neighbor frames.
        
        A detection is considered a false positive if:
        - The marker appears in frame i, but
        - The marker does NOT appear in ANY of the neighbor frames (within neighbor_num frames before/after)
        
        Args:
            detections: List of detection dictionaries with 'image_idx' and 'marker_id'
            neighbor_num: Number of neighbor frames to check on each side
            
        Returns:
            filtered_detections: List of detections with false positives removed
            removed_count: Number of false positive detections removed
        """
        if neighbor_num <= 0 or len(detections) == 0:
            return detections, 0
        
        # Build per-frame marker sets for quick lookup
        markers_by_frame = {}  # frame_idx -> set of marker_ids
        for det in detections:
            fidx = det['image_idx']
            mid = det['marker_id']
            if fidx not in markers_by_frame:
                markers_by_frame[fidx] = set()
            markers_by_frame[fidx].add(mid)
        
        # Get all frame indices (sorted)
        all_frame_indices = sorted(markers_by_frame.keys())
        if len(all_frame_indices) == 0:
            return detections, 0
        
        # Create mapping from frame index to position in sorted list
        frame_idx_to_pos = {fidx: pos for pos, fidx in enumerate(all_frame_indices)}
        
        # Filter detections
        filtered_detections = []
        removed_count = 0
        
        for det in detections:
            fidx = det['image_idx']
            mid = det['marker_id']
            
            # Get position of current frame in sorted list
            current_pos = frame_idx_to_pos[fidx]
            
            # Get neighbor frame positions (excluding current frame)
            neighbor_positions = []
            for offset in range(-neighbor_num, neighbor_num + 1):
                if offset == 0:
                    continue  # Skip current frame
                neighbor_pos = current_pos + offset
                if 0 <= neighbor_pos < len(all_frame_indices):
                    neighbor_fidx = all_frame_indices[neighbor_pos]
                    neighbor_positions.append(neighbor_fidx)
            
            # Check if marker appears in any neighbor frame
            is_false_positive = True
            if len(neighbor_positions) > 0:
                # Marker must appear in at least one neighbor frame
                for neighbor_fidx in neighbor_positions:
                    if mid in markers_by_frame[neighbor_fidx]:
                        is_false_positive = False
                        break
            else:
                # No neighbor frames available (at boundaries), keep the detection
                is_false_positive = False
            
            if is_false_positive:
                removed_count += 1
                print(f"[NO-BA2] Filtered false positive: marker {mid} in frame {fidx} "
                      f"(not present in {len(neighbor_positions)} neighbor frames)")
            else:
                filtered_detections.append(det)
        
        if removed_count > 0:
            print(f"[NO-BA2] Filtered {removed_count} false positive detections "
                  f"(using neighbor_num={neighbor_num})")
        
        return filtered_detections, removed_count

    def _build_detection_structures(self, detections):
        """
        Build basic data structures from detections.
        
        Returns:
            all_marker_ids: Sorted list of all marker IDs
            all_frame_indices: Sorted list of all frame indices
            n_frames: Number of frames
            detections_by_frame: Dictionary mapping frame_idx to list of detections
        """
        all_marker_ids = sorted(set(d['marker_id'] for d in detections))
        all_frame_indices = sorted(set(d['image_idx'] for d in detections))
        n_frames = len(all_frame_indices)
        
        detections_by_frame = {}
        for det in detections:
            fidx = det['image_idx']
            detections_by_frame.setdefault(fidx, []).append(det)
        
        return all_marker_ids, all_frame_indices, n_frames, detections_by_frame

    def _filter_markers_by_occurrence(self, detections, all_marker_ids):
        """
        Filter markers by occurrence count.
        
        Returns:
            filtered_detections: Detections with only valid markers
            valid_marker_ids: Set of valid marker IDs
            all_marker_ids: Updated sorted list of valid marker IDs
        """
        # Count how many times each marker appears
        marker_occurrence_count = {}
        for det in detections:
            marker_id = det['marker_id']
            marker_occurrence_count[marker_id] = marker_occurrence_count.get(marker_id, 0) + 1
        
        # Filter out markers that appear less than MIN_ARUCO_COUNT times
        valid_marker_ids = {mid for mid, count in marker_occurrence_count.items() 
                           if count >= MIN_ARUCO_COUNT}
        
        if not valid_marker_ids:
            return None, None, None
        
        # Filter detections to only include valid markers
        filtered_detections = [d for d in detections if d['marker_id'] in valid_marker_ids]
        removed_marker_count = len(all_marker_ids) - len(valid_marker_ids)
        
        if removed_marker_count > 0:
            removed_markers = [mid for mid in all_marker_ids if mid not in valid_marker_ids]
            print(f"[NO-BA2] Filtered out {removed_marker_count} markers that appear < {MIN_ARUCO_COUNT} times: {removed_markers}")
            for mid in removed_markers:
                count = marker_occurrence_count[mid]
                print(f"[NO-BA2]   Marker {mid}: {count} occurrences (threshold: {MIN_ARUCO_COUNT})")
        
        all_marker_ids = sorted(valid_marker_ids)
        print(f"[NO-BA2] After marker filtering: {len(valid_marker_ids)} valid markers, "
              f"{len(filtered_detections)} detections")
        
        return filtered_detections, valid_marker_ids, all_marker_ids

    def _filter_images_by_marker_count(self, detections, images, all_frame_indices):
        """
        Filter images with less than MIN_ARUCO_COUNT valid markers.
        
        Returns:
            filtered_detections: Detections with only valid frames
            filtered_images: Images with only valid frames
            all_frame_indices: Updated sorted list of valid frame indices
            original_kept_frame_indices: Original frame indices that were kept
        """
        original_kept_frame_indices = list(range(len(images)))
        
        # Count markers per frame
        frame_marker_count = {}
        for det in detections:
            fidx = det['image_idx']
            frame_marker_count[fidx] = frame_marker_count.get(fidx, 0) + 1
        
        # Find frames with sufficient markers
        valid_frame_indices = [fidx for fidx in all_frame_indices 
                              if frame_marker_count.get(fidx, 0) >= MIN_ARUCO_COUNT]
        
        if len(valid_frame_indices) < len(all_frame_indices):
            removed_frame_count = len(all_frame_indices) - len(valid_frame_indices)
            removed_frames = [fidx for fidx in all_frame_indices if fidx not in valid_frame_indices]
            print(f"[NO-BA2] Filtering out {removed_frame_count} images with < {MIN_ARUCO_COUNT} valid markers after false positive filtering")
            for fidx in removed_frames:
                count = frame_marker_count.get(fidx, 0)
                print(f"[NO-BA2]   Frame {fidx}: {count} valid markers (threshold: {MIN_ARUCO_COUNT})")
            
            # Filter detections to only include valid frames
            filtered_detections = [d for d in detections if d['image_idx'] in valid_frame_indices]
            
            # Update original_kept_frame_indices to only include valid frames
            original_kept_frame_indices = [fidx for fidx in original_kept_frame_indices if fidx in valid_frame_indices]
            
            # Create mapping from old frame index to new frame index
            old_to_new_idx = {old_fidx: new_idx for new_idx, old_fidx in enumerate(sorted(valid_frame_indices))}
            
            # Remap frame indices in detections
            for det in filtered_detections:
                det['image_idx'] = old_to_new_idx[det['image_idx']]
            
            # Filter images
            filtered_images = [images[old_fidx] for old_fidx in sorted(valid_frame_indices)]
            
            all_frame_indices = sorted(set(d['image_idx'] for d in filtered_detections))
            
            print(f"[NO-BA2] After image filtering: {len(filtered_images)} images, {len(filtered_detections)} detections, {len(all_frame_indices)} frames")
            
            return filtered_detections, filtered_images, all_frame_indices, original_kept_frame_indices
        
        return detections, images, all_frame_indices, original_kept_frame_indices

    def _select_world_frame_marker(self, all_marker_ids, all_frame_indices, detections_by_frame):
        """
        Select world frame marker based on co-occurrence with other markers.
        
        Returns:
            ref_marker_id: Selected reference marker ID
        """
        # Count co-occurrences: for each marker, count how many other markers appear with it in the same image
        marker_cooccurrence_scores = {}
        for mid in all_marker_ids:
            cooccurrence_count = 0
            for fidx in all_frame_indices:
                frame_dets = detections_by_frame.get(fidx, [])
                marker_ids_in_frame = set(d['marker_id'] for d in frame_dets)
                if mid in marker_ids_in_frame:
                    # Count how many OTHER markers appear with this marker in this frame
                    cooccurrence_count += len(marker_ids_in_frame) - 1  # -1 to exclude self
            marker_cooccurrence_scores[mid] = cooccurrence_count

        # Select marker with highest co-occurrence score
        ref_marker_id = max(marker_cooccurrence_scores, key=marker_cooccurrence_scores.get)
        print(f"[NO-BA2] Using marker {ref_marker_id} as world reference "
              f"(co-occurrence score: {marker_cooccurrence_scores[ref_marker_id]})")
        
        return ref_marker_id

    def _initialize_camera_poses_from_ref(self, all_frame_indices, detections_by_frame, ref_marker_id):
        """
        Initialize camera poses from reference marker.
        
        Returns:
            init_camera_poses: Dictionary mapping frame_idx to {rvec, tvec}
        """
        init_camera_poses = {}
        
        for fidx in all_frame_indices:
            frame_dets = detections_by_frame[fidx]
            ref_det = next((d for d in frame_dets if d['marker_id'] == ref_marker_id), None)
            if ref_det is not None:
                # Detection rvec,tvec are object->camera
                t_cm = np.array(ref_det['tvec'], dtype=np.float64).reshape(3)
                
                # world == ref marker, so:
                # R_wc = R_cm, t_wc = t_cm
                init_camera_poses[fidx] = {
                    'rvec': ref_det['rvec'].astype(np.float64).copy(),
                    'tvec': t_cm.copy()
                }
        
        return init_camera_poses

    def _initialize_marker_poses(self, all_marker_ids, all_frame_indices, detections_by_frame, ref_marker_id):
        """
        Initialize marker->world poses for non-ref markers (with outlier removal).
        
        Returns:
            init_marker_poses_mw: Dictionary mapping marker_id to {rvec, tvec}
        """
        init_marker_poses_mw = {}
        
        # Fix reference marker as identity in world frame
        init_marker_poses_mw[ref_marker_id] = {
            'rvec': np.zeros(3, dtype=np.float64),
            'tvec': np.zeros(3, dtype=np.float64)
        }

        # For each non-ref marker, use frames where both ref and this marker are visible
        for mid in all_marker_ids:
            if mid == ref_marker_id:
                continue

            R_mw_list = []
            t_mw_list = []

            for fidx in all_frame_indices:
                frame_dets = detections_by_frame[fidx]
                ref_det = next((d for d in frame_dets if d['marker_id'] == ref_marker_id), None)
                m_det = next((d for d in frame_dets if d['marker_id'] == mid), None)

                if ref_det is None or m_det is None:
                    continue

                # Ref marker detection: world->camera
                R_wc, _ = cv2.Rodrigues(ref_det['rvec'])
                t_wc = np.array(ref_det['tvec'], dtype=np.float64).reshape(3)

                # Marker m detection: marker->camera (object->camera)
                R_cm, _ = cv2.Rodrigues(m_det['rvec'])
                t_cm = np.array(m_det['tvec'], dtype=np.float64).reshape(3)

                # From equations:
                # R_cm = R_wc R_mw
                # t_cm = R_wc t_mw + t_wc
                # => R_mw = R_wc^T R_cm
                #    t_mw = R_wc^T (t_cm - t_wc)
                R_mw = R_wc.T @ R_cm
                t_mw = R_wc.T @ (t_cm - t_wc)

                R_mw_list.append(R_mw)
                t_mw_list.append(t_mw)

            if R_mw_list:
                # Remove outliers before averaging
                t_mw_array = np.array(t_mw_list)
                R_mw_array = np.array(R_mw_list)
                
                # Remove outliers from translations
                t_inlier_mask = self._remove_outliers_iqr(t_mw_array, axis=0)
                
                # Remove outliers from rotations
                R_inlier_mask = self._remove_outliers_iqr(R_mw_array, axis=1)
                
                # Combined mask: keep only poses that are inliers for both rotation and translation
                inlier_mask = t_inlier_mask & R_inlier_mask
                
                if np.sum(inlier_mask) < 2:
                    # If too few inliers, use all (better than nothing)
                    inlier_mask = np.ones(len(R_mw_list), dtype=bool)
                    print(f"[NO-BA2] Marker {mid}: too few inliers ({np.sum(inlier_mask)}), using all poses")
                else:
                    outlier_count = len(R_mw_list) - np.sum(inlier_mask)
                    if outlier_count > 0:
                        print(f"[NO-BA2] Marker {mid}: removed {outlier_count} outlier poses out of {len(R_mw_list)}")
                
                # Average only inlier poses
                R_mw_inliers = R_mw_array[inlier_mask]
                t_mw_inliers = t_mw_array[inlier_mask]
                
                # Average rotation via SVD, translation via mean
                R_avg = np.mean(R_mw_inliers, axis=0)
                U, _, Vt = np.linalg.svd(R_avg)
                R_avg = U @ Vt
                if np.linalg.det(R_avg) < 0:
                    R_avg = U @ np.diag([1, 1, -1]) @ Vt
                rvec_avg, _ = cv2.Rodrigues(R_avg)
                tvec_avg = np.mean(t_mw_inliers, axis=0)

                init_marker_poses_mw[mid] = {
                    'rvec': rvec_avg.flatten(),
                    'tvec': tvec_avg.flatten()
                }
                print(f"[NO-BA2] Marker {mid}: initialized from {np.sum(inlier_mask)} inlier poses "
                      f"(out of {len(R_mw_list)} co-visible frames with ref marker)")
            else:
                # Fallback: identity (not well-constrained but LSQ will adjust within its connected component)
                print(f"[NO-BA2] WARNING: Marker {mid} never co-visible with ref marker. Using identity init.")
                init_marker_poses_mw[mid] = {
                    'rvec': np.zeros(3, dtype=np.float64),
                    'tvec': np.zeros(3, dtype=np.float64)
                }
        
        return init_marker_poses_mw

    def _initialize_camera_poses_without_ref(self, all_frame_indices, detections_by_frame, 
                                             init_camera_poses, init_marker_poses_mw, 
                                             ref_marker_id, marker_size, reprojection_error_threshold):
        """
        Initialize camera poses for frames without ref marker (with reprojection error checking).
        
        Returns:
            init_camera_poses: Updated dictionary with all camera poses
            abandoned_frame_indices: Set of frame indices with high reprojection errors
        """
        obj_points = np.array([
            [-marker_size / 2,  marker_size / 2, 0],
            [ marker_size / 2,  marker_size / 2, 0],
            [ marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0]
        ], dtype=np.float32)

        abandoned_frame_indices = set()

        for fidx in all_frame_indices:
            if fidx in init_camera_poses:
                # Check reprojection error for frames initialized from ref marker
                frame_dets = detections_by_frame[fidx]
                object_points_3d = []
                image_points_2d = []
                
                for det in frame_dets:
                    mid = det['marker_id']
                    if mid == ref_marker_id:
                        # For ref marker, it's at origin in world frame
                        corners_3d_world = obj_points
                    elif mid in init_marker_poses_mw:
                        # marker->world
                        rvec_mw = init_marker_poses_mw[mid]['rvec']
                        tvec_mw = init_marker_poses_mw[mid]['tvec']
                        R_mw, _ = cv2.Rodrigues(rvec_mw)
                        t_mw = np.array(tvec_mw, dtype=np.float64).reshape(3)
                        corners_3d_world = (R_mw @ obj_points.T).T + t_mw
                    else:
                        continue
                    
                    corners_2d = det['corners']
                    object_points_3d.append(corners_3d_world)
                    image_points_2d.append(corners_2d)
                
                if object_points_3d:
                    object_points_3d = np.vstack(object_points_3d)
                    image_points_2d = np.vstack(image_points_2d)
                    
                    rvec_wc = init_camera_poses[fidx]['rvec']
                    tvec_wc = init_camera_poses[fidx]['tvec']
                    mean_error = self._compute_reprojection_error(
                        object_points_3d, image_points_2d, rvec_wc, tvec_wc, self.camera_intrinsic
                    )
                    
                    if mean_error > reprojection_error_threshold:
                        print(f"[NO-BA2] Frame {fidx}: abandoned due to high reprojection error "
                              f"({mean_error:.2f} pixels > {reprojection_error_threshold} pixels)")
                        abandoned_frame_indices.add(fidx)
                continue  # already initialized by ref marker

            frame_dets = detections_by_frame[fidx]

            object_points_3d = []
            image_points_2d = []

            for det in frame_dets:
                mid = det['marker_id']
                if mid not in init_marker_poses_mw:
                    continue

                # marker->world
                rvec_mw = init_marker_poses_mw[mid]['rvec']
                tvec_mw = init_marker_poses_mw[mid]['tvec']
                R_mw, _ = cv2.Rodrigues(rvec_mw)
                t_mw = np.array(tvec_mw, dtype=np.float64).reshape(3)

                # transform marker corners to world
                corners_3d_world = (R_mw @ obj_points.T).T + t_mw
                corners_2d = det['corners']
                object_points_3d.append(corners_3d_world)
                image_points_2d.append(corners_2d)

            if object_points_3d:
                object_points_3d = np.vstack(object_points_3d)
                image_points_2d = np.vstack(image_points_2d)

                success, rvec_wc, tvec_wc = cv2.solvePnP(
                    object_points_3d,
                    image_points_2d,
                    self.camera_intrinsic,
                    None,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    # Check reprojection error
                    mean_error = self._compute_reprojection_error(
                        object_points_3d, image_points_2d, rvec_wc, tvec_wc, self.camera_intrinsic
                    )
                    
                    if mean_error > reprojection_error_threshold:
                        print(f"[NO-BA2] Frame {fidx}: abandoned due to high reprojection error "
                              f"({mean_error:.2f} pixels > {reprojection_error_threshold} pixels)")
                        abandoned_frame_indices.add(fidx)
                    else:
                        init_camera_poses[fidx] = {
                            'rvec': rvec_wc.flatten(),
                            'tvec': tvec_wc.flatten()
                        }
                        print(f"[NO-BA2] Frame {fidx}: initialized camera pose from {len(frame_dets)} markers "
                              f"(reprojection error: {mean_error:.2f} pixels)")
                    continue

            # Fallback if nothing worked - also check if we should abandon
            print(f"[NO-BA2] WARNING: Frame {fidx}: no reliable init, using identity camera pose")
            init_camera_poses[fidx] = {
                'rvec': np.zeros(3, dtype=np.float64),
                'tvec': np.zeros(3, dtype=np.float64)
            }
        
        return init_camera_poses, abandoned_frame_indices

    def _filter_abandoned_frames(self, detections, images, init_camera_poses, 
                                 abandoned_frame_indices, original_kept_frame_indices):
        """
        Filter out abandoned frames from detections and images.
        
        Returns:
            filtered_detections: Detections with abandoned frames removed
            filtered_images: Images with abandoned frames removed
            filtered_init_camera_poses: Camera poses with abandoned frames removed
            all_frame_indices: Updated frame indices
            detections_by_frame: Updated detections_by_frame
            original_kept_frame_indices: Updated original kept frame indices
        """
        if not abandoned_frame_indices:
            print(f"[NO-BA2] No frames abandoned - all reprojection errors within threshold")
            all_frame_indices = sorted(set(d['image_idx'] for d in detections))
            detections_by_frame = {}
            for det in detections:
                fidx = det['image_idx']
                detections_by_frame.setdefault(fidx, []).append(det)
            return detections, images, init_camera_poses, all_frame_indices, detections_by_frame, original_kept_frame_indices
        
        print(f"[NO-BA2] Abandoning {len(abandoned_frame_indices)} frames due to high reprojection errors")
        # Filter detections
        filtered_detections = [d for d in detections if d['image_idx'] not in abandoned_frame_indices]
        # Filter images
        filtered_images = [img for idx, img in enumerate(images) if idx not in abandoned_frame_indices]
        
        # Update original_kept_frame_indices to exclude abandoned frames
        original_kept_frame_indices = [original_kept_frame_indices[idx] for idx in range(len(images)) 
                                       if idx not in abandoned_frame_indices]
        
        # Remap frame indices in detections
        old_to_new_idx = {}
        new_idx = 0
        for old_idx in range(len(images)):
            if old_idx not in abandoned_frame_indices:
                old_to_new_idx[old_idx] = new_idx
                new_idx += 1
        
        # Update detection frame indices
        for det in filtered_detections:
            det['image_idx'] = old_to_new_idx[det['image_idx']]
        
        # Update frame indices in init_camera_poses
        filtered_init_camera_poses = {}
        for old_fidx, pose in init_camera_poses.items():
            if old_fidx not in abandoned_frame_indices:
                new_fidx = old_to_new_idx[old_fidx]
                filtered_init_camera_poses[new_fidx] = pose
        
        all_frame_indices = sorted(set(d['image_idx'] for d in filtered_detections))
        
        # Rebuild detections_by_frame with new indices
        detections_by_frame = {}
        for det in filtered_detections:
            fidx = det['image_idx']
            detections_by_frame.setdefault(fidx, []).append(det)
        
        print(f"[NO-BA2] After filtering: {len(filtered_detections)} detections, {len(filtered_images)} images, {len(all_frame_indices)} frames")
        
        return filtered_detections, filtered_images, filtered_init_camera_poses, all_frame_indices, detections_by_frame, original_kept_frame_indices

    @staticmethod
    def _rvec_to_quat(rvec):
        """Convert rotation vector to quaternion (w, x, y, z)"""
        angle = np.linalg.norm(rvec)
        if angle < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        axis = rvec / angle
        half_angle = angle / 2.0
        s = np.sin(half_angle)
        return np.array([np.cos(half_angle), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64)

    @staticmethod
    def _quat_to_rot_matrix(q):
        """Convert quaternion (w, x, y, z) to rotation matrix (3x3)"""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        # Normalize quaternion
        norm = torch.sqrt(w*w + x*x + y*y + z*z + 1e-8)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Build rotation matrix
        R = torch.stack([
            torch.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], dim=-1),
            torch.stack([2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)], dim=-1),
            torch.stack([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)], dim=-1)
        ], dim=-2)
        return R

    @staticmethod
    def _quat_to_rvec(quat):
        """Convert quaternion (w, x, y, z) to rotation vector"""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Normalize
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-8:
            return np.zeros(3, dtype=np.float64)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix then to rvec
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(R)
        return rvec.flatten()

    def _prepare_pytorch_optimization(self, detections, all_marker_ids, all_frame_indices, 
                                     init_marker_poses_mw, init_camera_poses, ref_marker_id, marker_size):
        """
        Prepare data structures for PyTorch optimization.
        
        Returns:
            detected_corners_t: Tensor of detected corners
            visibility_mask_t: Tensor of visibility mask
            K_t: Camera intrinsic tensor
            obj_points_t: Object points tensor
            marker_params_t: Marker parameters tensor
            camera_params_t: Camera parameters tensor
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler (ExponentialLR)
            marker_id_to_idx: Mapping from marker ID to index
            frame_idx_to_param_idx: Mapping from frame index to parameter index
            free_marker_ids: List of free marker IDs
            n_images: Number of images
            n_aruco: Number of ArUco markers
            n_corners: Number of corners per marker
        """
        free_marker_ids = [mid for mid in all_marker_ids if mid != ref_marker_id]
        n_free_markers = len(free_marker_ids)
        
        marker_id_to_free_idx = {mid: i for i, mid in enumerate(free_marker_ids)}
        frame_idx_to_param_idx = {fidx: i for i, fidx in enumerate(all_frame_indices)}
        
        K = self.camera_intrinsic.astype(np.float64)
        
        # Build batched tensors: (n_images, n_aruco, n_corners, 2)
        n_images = len(all_frame_indices)
        n_aruco = len(all_marker_ids)
        n_corners = 4  # ArUco markers have 4 corners
        
        # Create marker_id to index mapping (including ref marker)
        marker_id_to_idx = {mid: i for i, mid in enumerate(all_marker_ids)}
        
        # Initialize batched tensors
        detected_corners = np.zeros((n_images, n_aruco, n_corners, 2), dtype=np.float64)
        visibility_mask = np.zeros((n_images, n_aruco, 1), dtype=np.float64)
        
        # Fill detected corners and visibility mask
        for det in detections:
            mid = det['marker_id']
            fidx = det['image_idx']
            frame_idx = frame_idx_to_param_idx[fidx]
            marker_idx = marker_id_to_idx[mid]
            
            corners = det['corners'].reshape(n_corners, 2)
            detected_corners[frame_idx, marker_idx, :, :] = corners
            visibility_mask[frame_idx, marker_idx, 0] = 1.0
        
        # Initialize PyTorch parameters (quaternion + translation = 7 params)
        # Marker poses: (n_free_markers, 7) [quat(4), tvec(3)]
        marker_params = []
        for mid in free_marker_ids:
            rvec = init_marker_poses_mw[mid]['rvec']
            tvec = init_marker_poses_mw[mid]['tvec']
            quat = self._rvec_to_quat(rvec)
            marker_params.append(np.concatenate([quat, tvec]))
        marker_params = np.array(marker_params, dtype=np.float32)
        
        # Camera poses: (n_frames, 7) [quat(4), tvec(3)]
        camera_params = []
        for fidx in all_frame_indices:
            rvec = init_camera_poses[fidx]['rvec']
            tvec = init_camera_poses[fidx]['tvec']
            quat = self._rvec_to_quat(rvec)
            camera_params.append(np.concatenate([quat, tvec]))
        camera_params = np.array(camera_params, dtype=np.float32)
        
        # Object points
        obj_points = np.array([
            [-marker_size / 2,  marker_size / 2, 0],
            [ marker_size / 2,  marker_size / 2, 0],
            [ marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0]
        ], dtype=np.float32)
        
        # Convert to PyTorch tensors
        device = torch.device('cpu')
        detected_corners_t = torch.from_numpy(detected_corners).to(device)
        visibility_mask_t = torch.from_numpy(visibility_mask).to(device)
        K_t = torch.from_numpy(K).to(device)
        obj_points_t = torch.from_numpy(obj_points).to(device)
        
        marker_params_t = torch.nn.Parameter(torch.from_numpy(marker_params).to(device), requires_grad=True)
        camera_params_t = torch.nn.Parameter(torch.from_numpy(camera_params).to(device), requires_grad=True)
        
        optimizer = torch.optim.Adam([marker_params_t, camera_params_t], lr=BA_LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=BA_DECAY_RATE)
        
        return (detected_corners_t, visibility_mask_t, K_t, obj_points_t, 
                marker_params_t, camera_params_t, optimizer, scheduler,
                marker_id_to_idx, frame_idx_to_param_idx, free_marker_ids,
                n_images, n_aruco, n_corners)

    def _run_pytorch_optimization(self, detected_corners_t, visibility_mask_t, K_t, obj_points_t,
                                 marker_params_t, camera_params_t, optimizer, scheduler,
                                 marker_id_to_idx, frame_idx_to_param_idx, free_marker_ids,
                                 ref_marker_id, n_images, n_aruco, n_corners, enable_ba):
        """
        Run PyTorch optimization loop.
        
        Returns:
            marker_params_opt: Optimized marker parameters
            camera_params_opt: Optimized camera parameters
            final_mean_error: Final mean reprojection error
        """
        print(f"[NO-BA2] Starting PyTorch optimization with "
              f"{len(free_marker_ids) * 7 + n_images * 7} parameters "
              f"({len(free_marker_ids)} markers, {n_images} frames)...")
        start_t = time.time()
        
        device = detected_corners_t.device
        
        # Optimization loop
        max_iterations = 1 if not enable_ba else BA_MAX_ITERATIONS
        for iter_idx in range(max_iterations):
            optimizer.zero_grad()
            
            # Get marker poses (marker->world)
            # Build full marker pose tensor including ref marker
            marker_quats_all = torch.zeros((n_aruco, 4), device=device, dtype=torch.float32)
            marker_tvecs_all = torch.zeros((n_aruco, 3), device=device, dtype=torch.float32)
            
            # Ref marker is fixed at identity
            ref_marker_idx = marker_id_to_idx[ref_marker_id]
            marker_quats_all[ref_marker_idx] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
            marker_tvecs_all[ref_marker_idx] = torch.zeros(3, device=device, dtype=torch.float32)
            
            # Free markers from parameters
            for i, mid in enumerate(free_marker_ids):
                marker_idx = marker_id_to_idx[mid]
                marker_quats_all[marker_idx] = marker_params_t[i, :4]
                marker_tvecs_all[marker_idx] = marker_params_t[i, 4:7]
            
            # Get camera poses (world->camera)
            camera_quats = camera_params_t[:, :4]
            camera_tvecs = camera_params_t[:, 4:7]
            
            # Convert quaternions to rotation matrices
            # marker->world: (n_aruco, 3, 3)
            R_mw = self._quat_to_rot_matrix(marker_quats_all.unsqueeze(0)).squeeze(0)
            # world->camera: (n_images, 3, 3)
            R_wc = self._quat_to_rot_matrix(camera_quats)
            
            # Compute marker->camera: R_cm = R_wc @ R_mw, t_cm = R_wc @ t_mw + t_wc
            # R_cm: (n_images, n_aruco, 3, 3)
            R_wc_expanded = R_wc.unsqueeze(1)  # (n_images, 1, 3, 3)
            R_mw_expanded = R_mw.unsqueeze(0)  # (1, n_aruco, 3, 3)
            R_cm = R_wc_expanded @ R_mw_expanded  # (n_images, n_aruco, 3, 3)
            
            # t_cm: (n_images, n_aruco, 3)
            t_mw_expanded = marker_tvecs_all.unsqueeze(0)  # (1, n_aruco, 3)
            t_cm = (R_wc_expanded @ t_mw_expanded.unsqueeze(-1)).squeeze(-1)  # (n_images, n_aruco, 3)
            t_cm = t_cm + camera_tvecs.unsqueeze(1)  # (n_images, n_aruco, 3)
            
            # Transform marker corners to camera frame
            # obj_points: (4, 3), expand to (n_images, n_aruco, 4, 3)
            obj_points_expanded = obj_points_t.unsqueeze(0).unsqueeze(0).expand(n_images, n_aruco, -1, -1)
            # corners_3d_cam: (n_images, n_aruco, 4, 3)
            corners_3d_cam = torch.matmul(obj_points_expanded, R_cm.transpose(-2, -1)) + t_cm.unsqueeze(2)
            
            # Project to 2D
            # corners_3d_cam: (n_images, n_aruco, 4, 3)
            fx, fy = K_t[0, 0], K_t[1, 1]
            cx, cy = K_t[0, 2], K_t[1, 2]
            
            x_cam = corners_3d_cam[..., 0]
            y_cam = corners_3d_cam[..., 1]
            z_cam = corners_3d_cam[..., 2] + 1e-8  # Avoid division by zero
            
            u = fx * x_cam / z_cam + cx
            v = fy * y_cam / z_cam + cy
            projected_corners = torch.stack([u, v], dim=-1)  # (n_images, n_aruco, 4, 2)
            
            # Compute reprojection error
            error = (detected_corners_t - projected_corners) * visibility_mask_t.unsqueeze(-1)  # (n_images, n_aruco, 4, 2)
            error_norm = torch.norm(error, dim=-1, keepdim=True)  # (n_images, n_aruco, 4, 1)
            # visibility_mask_t: (n_images, n_aruco, 1), expands to (n_images, n_aruco, 4, 1)
            mask_expanded = visibility_mask_t.unsqueeze(-2)  # (n_images, n_aruco, 1, 1)
            loss = torch.sum(error_norm ** 2 * mask_expanded) / (torch.sum(mask_expanded) * n_corners + 1e-8)

            if enable_ba:
                loss.backward()
                optimizer.step()
                # Apply learning rate decay
                scheduler.step()
            else:
                # Clear gradients without backward propagation
                optimizer.zero_grad()
            
            # Log every 10 iterations (more frequent updates)
            if (iter_idx + 1) % 10 == 0 or iter_idx == 0:
                # Compute mean error over visible corners only
                reprojection_error = error_norm * mask_expanded
                mean_error = reprojection_error.sum() / (mask_expanded.sum() * n_corners + 1e-8)
                mean_error = mean_error.item()
                progress_pct = int(100 * (iter_idx + 1) / max_iterations)
                current_lr = optimizer.param_groups[0]['lr']
                log_msg = f"Optimizing... Iter {iter_idx+1}/{max_iterations} ({progress_pct}%): error = {mean_error:.2f} pixels, lr = {current_lr:.6f}, decay = {BA_DECAY_RATE}"
                print(f"[NO-BA2] Iter {iter_idx+1}/{max_iterations}: mean reprojection error = {mean_error:.4f} pixels, lr = {current_lr:.6f}, decay = {BA_DECAY_RATE}")
                if self.progress_callback:
                    self.progress_callback(log_msg, iter_idx + 1, max_iterations)

        reprojection_error = error_norm * mask_expanded
        mean_error = reprojection_error.sum() / (mask_expanded.sum() * n_corners + 1e-8)
        mean_error = mean_error.item()
        
        end_t = time.time()
        print(f"[NO-BA2] Optimization done. Time: {end_t - start_t:.2f}s")
        
        # Convert back to numpy
        marker_params_opt = marker_params_t.detach().cpu().numpy()
        camera_params_opt = camera_params_t.detach().cpu().numpy()
        
        return marker_params_opt, camera_params_opt, mean_error

    def _decode_optimized_parameters(self, marker_params_opt, camera_params_opt, 
                                     free_marker_ids, all_frame_indices, ref_marker_id):
        """
        Decode optimized parameters back to marker and camera poses.
        
        Returns:
            marker_poses_world: Dictionary mapping marker_id to {rvec, tvec}
            camera_poses: Dictionary mapping frame_idx to {rvec, tvec}
        """
        # Marker poses (marker->world)
        marker_poses_world = {}

        # Reference marker fixed
        marker_poses_world[ref_marker_id] = {
            'rvec': np.zeros(3, dtype=np.float64),
            'tvec': np.zeros(3, dtype=np.float64)
        }

        # Non-ref from optimized params
        for i, mid in enumerate(free_marker_ids):
            quat = marker_params_opt[i, :4]
            tvec = marker_params_opt[i, 4:7]
            rvec = self._quat_to_rvec(quat)
            marker_poses_world[mid] = {
                'rvec': rvec,
                'tvec': tvec.astype(np.float64)
            }

        # Camera poses (world->camera)
        camera_poses = {}
        for i, fidx in enumerate(all_frame_indices):
            quat = camera_params_opt[i, :4]
            tvec = camera_params_opt[i, 4:7]
            rvec = self._quat_to_rvec(quat)
            camera_poses[fidx] = {
                'rvec': rvec,
                'tvec': tvec.astype(np.float64)
            }
        
        return marker_poses_world, camera_poses

    def _update_detections_with_poses(self, detections, marker_poses_world, camera_poses):
        """
        Update detections with optimized marker->camera poses.
        
        Returns:
            updated_detections: List of detections with updated poses
        """
        updated_detections = []
        for det in detections:
            mid = det['marker_id']
            fidx = det['image_idx']

            # marker->world
            rvec_mw = marker_poses_world[mid]['rvec']
            tvec_mw = marker_poses_world[mid]['tvec']
            R_mw, _ = cv2.Rodrigues(rvec_mw)
            t_mw = np.array(tvec_mw, dtype=np.float64).reshape(3)

            # world->camera
            rvec_wc = camera_poses[fidx]['rvec']
            tvec_wc = camera_poses[fidx]['tvec']
            R_wc, _ = cv2.Rodrigues(rvec_wc)
            t_wc = np.array(tvec_wc, dtype=np.float64).reshape(3)

            # marker->camera
            R_cm = R_wc @ R_mw
            t_cm = R_wc @ t_mw + t_wc
            rvec_cm, _ = cv2.Rodrigues(R_cm)

            new_det = det.copy()
            new_det['rvec'] = rvec_cm.flatten()
            new_det['tvec'] = t_cm.flatten()
            updated_detections.append(new_det)
        
        return updated_detections

    def _compute_and_print_reprojection_errors(self, updated_detections, marker_size):
        """
        Compute and print final reprojection errors.
        
        Returns:
            marker_errors: Dictionary mapping marker_id to list of errors
            frame_errors: Dictionary mapping frame_idx to list of errors
        """
        obj_points = np.array([
            [-marker_size / 2,  marker_size / 2, 0],
            [ marker_size / 2,  marker_size / 2, 0],
            [ marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0]
        ], dtype=np.float32)
        
        print("\n[NO-BA2] ========== ArUco Marker Pose Errors ==========")
        
        # Per-marker errors
        marker_errors = {}  # marker_id -> list of errors
        frame_errors = {}   # frame_idx -> list of errors
        
        for det in updated_detections:
            mid = det['marker_id']
            fidx = det['image_idx']
            
            # Get optimized marker->camera pose
            rvec_cm = det['rvec']
            tvec_cm = det['tvec']
            
            # Project marker corners
            projected, _ = cv2.projectPoints(
                obj_points,
                rvec_cm,
                tvec_cm,
                self.camera_intrinsic,
                None
            )
            projected = projected.reshape(-1, 2)
            
            # Observed corners
            corners_2d = det['corners'].reshape(-1, 2)
            
            # Compute per-corner errors
            errors = np.linalg.norm(corners_2d - projected, axis=1)
            mean_error = np.mean(errors)
            
            # Store errors
            if mid not in marker_errors:
                marker_errors[mid] = []
            marker_errors[mid].append(mean_error)
            
            if fidx not in frame_errors:
                frame_errors[fidx] = []
            frame_errors[fidx].append(mean_error)
        
        # Compute overall statistics first for outlier threshold
        all_errors = [err for errors in marker_errors.values() for err in errors]
        if all_errors:
            overall_mean = np.mean(all_errors)
            overall_std = np.std(all_errors)
            # Outlier threshold: mean + 2*std or 10 pixels, whichever is higher
            outlier_threshold = max(overall_mean + 2 * overall_std, 10.0)
        else:
            outlier_threshold = 10.0
        
        # Print per-marker statistics
        print("\n[NO-BA2] Per-Marker Reprojection Errors (pixels):")
        for mid in sorted(marker_errors.keys()):
            errors = marker_errors[mid]
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            min_err = np.min(errors)
            max_err = np.max(errors)
            # Count outliers (errors > threshold)
            n_outliers = sum(1 for e in errors if e > outlier_threshold)
            print(f"  Marker {mid:3d}: mean={mean_err:6.2f}, std={std_err:6.2f}, "
                  f"min={min_err:6.2f}, max={max_err:6.2f}, outliers={n_outliers:2d} "
                  f"({len(errors)} detections)")
        
        # Print per-frame statistics
        print("\n[NO-BA2] Per-Frame Reprojection Errors (pixels):")
        for fidx in sorted(frame_errors.keys()):
            errors = frame_errors[fidx]
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            min_err = np.min(errors)
            max_err = np.max(errors)
            # Count outliers (errors > threshold)
            n_outliers = sum(1 for e in errors if e > outlier_threshold)
            print(f"  Frame {fidx:3d}: mean={mean_err:6.2f}, std={std_err:6.2f}, "
                  f"min={min_err:6.2f}, max={max_err:6.2f}, outliers={n_outliers:2d} "
                  f"({len(errors)} markers)")
        
        # Print overall statistics
        if all_errors:
            overall_min = np.min(all_errors)
            overall_max = np.max(all_errors)
            # Count overall outliers
            n_total_outliers = sum(1 for e in all_errors if e > outlier_threshold)
            print(f"\n[NO-BA2] Overall Statistics:")
            print(f"  Mean error: {overall_mean:.2f} pixels")
            print(f"  Std error:  {overall_std:.2f} pixels")
            print(f"  Min error:  {overall_min:.2f} pixels")
            print(f"  Max error:  {overall_max:.2f} pixels")
            print(f"  Outlier threshold: {outlier_threshold:.2f} pixels")
            print(f"  Total outliers: {n_total_outliers} out of {len(all_errors)} detections "
                  f"({100.0 * n_total_outliers / len(all_errors):.1f}%)")
        
        print("[NO-BA2] ================================================\n")
        
        return marker_errors, frame_errors

    def _filter_frames_by_reprojection_error(self, updated_detections, images, camera_poses,
                                             all_frame_indices, frame_errors, 
                                             original_kept_frame_indices, reprojection_error_threshold):
        """
        Filter frames based on reprojection error threshold.
        
        Returns:
            filtered_detections: Detections with high-error frames removed
            filtered_images: Images with high-error frames removed
            filtered_camera_poses: Camera poses with high-error frames removed
            filtered_original_indices: Updated original kept frame indices
        """
        # Calculate mean reprojection error per frame
        frame_mean_errors = {}
        for fidx in frame_errors.keys():
            errors = frame_errors[fidx]
            mean_err = np.mean(errors)
            frame_mean_errors[fidx] = mean_err
        
        # Find frames with mean error > threshold
        filtered_out_frames = set()
        for fidx, mean_err in frame_mean_errors.items():
            if mean_err > reprojection_error_threshold:
                filtered_out_frames.add(fidx)
                print(f"[NO-BA2] Filtering out frame {fidx}: mean reprojection error = {mean_err:.2f} pixels > {reprojection_error_threshold} pixels")
        
        # Filter out frames with high reprojection error
        if filtered_out_frames:
            print(f"[NO-BA2] Filtering out {len(filtered_out_frames)} frames due to high reprojection error (threshold: {reprojection_error_threshold} pixels)")
            
            # Filter detections
            filtered_detections = [d for d in updated_detections if d['image_idx'] not in filtered_out_frames]
            
            # Filter images - need to map frame indices back to image indices
            # all_frame_indices contains the frame indices used in optimization
            # We need to find which image indices correspond to filtered frames
            filtered_image_indices = set()
            for fidx in filtered_out_frames:
                # Find the position of fidx in all_frame_indices
                if fidx in all_frame_indices:
                    pos = all_frame_indices.index(fidx)
                    if pos < len(images):
                        filtered_image_indices.add(pos)
            
            filtered_images = [img for idx, img in enumerate(images) if idx not in filtered_image_indices]
            
            # Update original_kept_frame_indices to exclude filtered frames
            # Map filtered frame indices to original image indices
            frame_to_original_idx = {}
            for i, fidx in enumerate(all_frame_indices):
                if i < len(original_kept_frame_indices):
                    frame_to_original_idx[fidx] = original_kept_frame_indices[i]
            
            # Remove original indices corresponding to filtered frames
            filtered_original_indices = []
            for fidx in all_frame_indices:
                if fidx not in filtered_out_frames and fidx in frame_to_original_idx:
                    filtered_original_indices.append(frame_to_original_idx[fidx])
            
            # Remap frame indices in detections
            old_to_new_idx = {}
            new_idx = 0
            for old_fidx in sorted(all_frame_indices):
                if old_fidx not in filtered_out_frames:
                    old_to_new_idx[old_fidx] = new_idx
                    new_idx += 1
            
            # Update detection frame indices
            for det in filtered_detections:
                det['image_idx'] = old_to_new_idx[det['image_idx']]
            
            # Filter camera poses
            filtered_camera_poses = {}
            for old_fidx, pose in camera_poses.items():
                if old_fidx not in filtered_out_frames:
                    new_fidx = old_to_new_idx[old_fidx]
                    filtered_camera_poses[new_fidx] = pose
            
            print(f"[NO-BA2] After filtering: {len(filtered_detections)} detections, {len(filtered_images)} images, {len(filtered_camera_poses)} camera poses")
            
            return filtered_detections, filtered_images, filtered_camera_poses, filtered_original_indices
        else:
            print(f"[NO-BA2] No frames filtered - all frames have mean reprojection error <= {reprojection_error_threshold} pixels")
            return updated_detections, images, camera_poses, original_kept_frame_indices

    def bundle_adjustment(self, detections, images, marker_size, enable_ba=True):
        """
        Lightweight joint optimization of marker poses (in world frame) and camera poses.

        World frame definition:
            - Choose the ArUco marker that co-occurs with other markers in the same image as much as possible.
            - Its pose in the world is fixed to identity (R = I, t = 0).

        Improvements:
            1. World frame marker selection based on co-occurrence with other markers
            2. Outlier removal when averaging marker poses across frames
            3. Reprojection error checking - abandon images with high errors (>10 pixels)
            4. Filter detections and images to exclude abandoned ones
            5. Return filtered images

        Parameterization:
            - For each non-reference marker m:
                marker->world pose: rvec_mw(3) + tvec_mw(3)
            - For each frame f:
                world->camera pose: rvec_wc(3) + tvec_wc(3)

        Residuals:
            - For each detection (frame f, marker m), we project the 3D marker corners through:
                X_world = R_mw @ X_marker + t_mw
                X_cam   = R_wc @ X_world + t_wc
              and compare with the observed 2D corners.
        """
        REPROJECTION_ERROR_THRESHOLD = 100.0  # pixels
        
        # Empty result structure
        empty_result = {
            'detections': [],
            'camera_intrinsic': self.camera_intrinsic,
            'camera_poses': {},
            'marker_poses_world': {},
            'images': [],
            'kept_frame_indices': []
        }
        
        # 0) Filter false positive detections using neighbor frames
        if ISOLATION_NEIGHBOR_NUM > 0:
            print(f"[NO-BA2] Filtering false positive detections using neighbor_num={ISOLATION_NEIGHBOR_NUM}...")
            detections, removed_fp_count = self._filter_false_positive_detections(detections, ISOLATION_NEIGHBOR_NUM)
            
            if not detections:
                print(f"[NO-BA2] ERROR: All detections filtered as false positives")
                return empty_result
            
            print(f"[NO-BA2] After false positive filtering: {len(detections)} detections")
        
        # Build basic data structures
        all_marker_ids, all_frame_indices, n_frames, detections_by_frame = self._build_detection_structures(detections)
        
        # 1) Filter markers by occurrence count
        filtered_detections, valid_marker_ids, all_marker_ids = self._filter_markers_by_occurrence(detections, all_marker_ids)
        if filtered_detections is None:
            print(f"[NO-BA2] ERROR: No markers appear at least {MIN_ARUCO_COUNT} times")
            return empty_result
        
        detections = filtered_detections
        all_marker_ids, all_frame_indices, n_frames, detections_by_frame = self._build_detection_structures(detections)
        
        # 1.5) Filter images with less than MIN_ARUCO_COUNT valid markers
        detections, images, all_frame_indices, original_kept_frame_indices = self._filter_images_by_marker_count(
            detections, images, all_frame_indices
        )
        
        if not detections or not images:
            print(f"[NO-BA2] ERROR: No valid images or detections after filtering")
            return empty_result
        
        all_marker_ids, all_frame_indices, n_frames, detections_by_frame = self._build_detection_structures(detections)
        
        # 2) Determine world frame marker
        ref_marker_id = self._select_world_frame_marker(all_marker_ids, all_frame_indices, detections_by_frame)
        
        # 3) Initial camera poses from reference marker
        init_camera_poses = self._initialize_camera_poses_from_ref(
            all_frame_indices, detections_by_frame, ref_marker_id
        )
        
        # 4) Initial marker->world poses for non-ref markers
        init_marker_poses_mw = self._initialize_marker_poses(
            all_marker_ids, all_frame_indices, detections_by_frame, ref_marker_id
        )
        
        # 5) Initial camera poses for frames without ref marker
        init_camera_poses, abandoned_frame_indices = self._initialize_camera_poses_without_ref(
            all_frame_indices, detections_by_frame, init_camera_poses, init_marker_poses_mw,
            ref_marker_id, marker_size, REPROJECTION_ERROR_THRESHOLD
        )
        
        # 6) Filter out abandoned frames
        detections, images, init_camera_poses, all_frame_indices, detections_by_frame, original_kept_frame_indices = \
            self._filter_abandoned_frames(
                detections, images, init_camera_poses, abandoned_frame_indices, original_kept_frame_indices
            )
        
        all_marker_ids, all_frame_indices, n_frames, detections_by_frame = self._build_detection_structures(detections)
        
        # 7) Prepare for PyTorch optimization
        (detected_corners_t, visibility_mask_t, K_t, obj_points_t,
         marker_params_t, camera_params_t, optimizer, scheduler,
         marker_id_to_idx, frame_idx_to_param_idx, free_marker_ids,
         n_images, n_aruco, n_corners) = self._prepare_pytorch_optimization(
            detections, all_marker_ids, all_frame_indices,
            init_marker_poses_mw, init_camera_poses, ref_marker_id, marker_size
        )
        
        # 8) Run PyTorch optimization
        marker_params_opt, camera_params_opt, final_mean_error = self._run_pytorch_optimization(
            detected_corners_t, visibility_mask_t, K_t, obj_points_t,
            marker_params_t, camera_params_t, optimizer, scheduler,
            marker_id_to_idx, frame_idx_to_param_idx, free_marker_ids,
            ref_marker_id, n_images, n_aruco, n_corners, enable_ba
        )
        
        # 9) Decode optimized parameters
        marker_poses_world, camera_poses = self._decode_optimized_parameters(
            marker_params_opt, camera_params_opt, free_marker_ids, all_frame_indices, ref_marker_id
        )
        
        # 10) Update detections with optimized poses
        updated_detections = self._update_detections_with_poses(
            detections, marker_poses_world, camera_poses
        )
        
        # 11) Compute and print final reprojection errors
        marker_errors, frame_errors = self._compute_and_print_reprojection_errors(
            updated_detections, marker_size
        )
        
        # 12) Filter frames based on reprojection error threshold
        updated_detections, images, camera_poses, original_kept_frame_indices = \
            self._filter_frames_by_reprojection_error(
                updated_detections, images, camera_poses, all_frame_indices, frame_errors,
                original_kept_frame_indices, REPROJECTION_ERROR_THRESHOLD
            )
        
        print("[NO-BA2] Completed pose estimation with lightweight optimization")

        return {
            'detections': updated_detections,
            'camera_intrinsic': self.camera_intrinsic,
            'camera_poses': camera_poses,          # world->camera
            'marker_poses_world': marker_poses_world,  # marker->world
            'images': images,  # Return filtered images (without abandoned ones)
            'kept_frame_indices': original_kept_frame_indices  # Original frame indices that were kept (for mapping image_paths)
        }

