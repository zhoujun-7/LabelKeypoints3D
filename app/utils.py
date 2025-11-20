"""
Utility functions for safe operations, path validation, and common tasks
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2


def safe_join_path(base_dir: str, filename: str) -> Path:
    """
    Safely join paths, preventing directory traversal attacks.
    
    Args:
        base_dir: Base directory path
        filename: Filename to join (must not contain path traversal)
    
    Returns:
        Path object for the joined path
    
    Raises:
        ValueError: If path traversal is detected
    """
    base_path = Path(base_dir).resolve()
    file_path = Path(filename)
    
    # Prevent path traversal
    if '..' in file_path.parts or file_path.is_absolute():
        raise ValueError(f"Path traversal detected or absolute path: {filename}")
    
    full_path = (base_path / file_path).resolve()
    
    # Ensure the result is within base_dir
    try:
        full_path.relative_to(base_path)
    except ValueError:
        raise ValueError(f"Path traversal detected: {filename}")
    
    return full_path


def safe_read_json(json_path: str) -> Dict[str, Any]:
    """
    Safely read JSON file with proper error handling.
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        Parsed JSON data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be read
        ValueError: If JSON is invalid
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    if not json_path.is_file():
        raise ValueError(f"Path is not a file: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except PermissionError as e:
        raise PermissionError(f"Permission denied reading {json_path}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_path}: {e}")


def safe_write_json(data: Any, json_path: str, indent: int = 2) -> None:
    """
    Safely write JSON file with proper error handling.
    
    Args:
        data: Data to write
        json_path: Path to JSON file
        indent: JSON indentation
    
    Raises:
        PermissionError: If file cannot be written
        OSError: If directory doesn't exist or other OS error
    """
    json_path = Path(json_path)
    
    # Create parent directory if it doesn't exist
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
    except PermissionError as e:
        raise PermissionError(f"Permission denied writing {json_path}: {e}")
    except OSError as e:
        raise OSError(f"Failed to write {json_path}: {e}")


def validate_frame_id(frame_id: int, num_frames: int) -> bool:
    """
    Validate that frame_id is within valid range.
    
    Args:
        frame_id: Frame ID to validate
        num_frames: Total number of frames
    
    Returns:
        True if valid, False otherwise
    """
    return 0 <= frame_id < num_frames


def get_image_name_safe(image_paths: List[str], frame_id: int) -> str:
    """
    Safely get image name for frame_id with bounds checking.
    
    Args:
        image_paths: List of image paths
        frame_id: Frame ID
    
    Returns:
        Image filename
    
    Raises:
        IndexError: If frame_id is out of range
    """
    if not validate_frame_id(frame_id, len(image_paths)):
        raise IndexError(
            f"Frame ID {frame_id} out of range [0, {len(image_paths)})"
        )
    return os.path.basename(image_paths[frame_id])


def safe_get_nested_dict(data: Dict, *keys, default: Any = None) -> Any:
    """
    Safely get nested dictionary value.
    
    Args:
        data: Dictionary to access
        *keys: Keys to traverse
        default: Default value if key not found
    
    Returns:
        Value at nested key or default
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def validate_image_dimensions(img: np.ndarray) -> bool:
    """
    Validate image dimensions are within safe limits.
    
    Args:
        img: Image array
    
    Returns:
        True if dimensions are valid
    """
    if img is None:
        return False
    h, w = img.shape[:2]
    return 0 < w <= MAX_IMAGE_DIMENSION and 0 < h <= MAX_IMAGE_DIMENSION


def calculate_bbox_safe(keypoints_2d: List[List[float]]) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculate bounding box from keypoints with validation.
    
    Args:
        keypoints_2d: List of [x, y] keypoint coordinates
    
    Returns:
        (x_min, y_min, x_max, y_max) or None if invalid
    """
    from .constants import BBOX_PADDING_RATIO, MIN_BBOX_DIMENSION
    
    if not keypoints_2d:
        return None
    
    try:
        xs = [kp[0] for kp in keypoints_2d if len(kp) >= 2]
        ys = [kp[1] for kp in keypoints_2d if len(kp) >= 2]
        
        if not xs or not ys:
            return None
        
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Check minimum dimensions
        if width < MIN_BBOX_DIMENSION or height < MIN_BBOX_DIMENSION:
            return None
        
        # Add padding
        padding_x = width * BBOX_PADDING_RATIO
        padding_y = height * BBOX_PADDING_RATIO
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = x_max + padding_x
        y_max = y_max + padding_y
        
        return (x_min, y_min, x_max, y_max)
    except (ValueError, TypeError, IndexError):
        return None


def triangulate_points(pt1: np.ndarray, pt2: np.ndarray, 
                      P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Triangulate 3D point from two 2D observations.
    
    This is a shared utility function to avoid code duplication.
    
    Args:
        pt1: 2D point in first image [x, y]
        pt2: 2D point in second image [x, y]
        P1: 3x4 projection matrix for first camera
        P2: 3x4 projection matrix for second camera
    
    Returns:
        3D point [x, y, z] in world coordinates
    
    Raises:
        ValueError: If projection matrices are invalid
    """
    # Validate inputs
    if pt1.shape != (2,) or pt2.shape != (2,):
        raise ValueError("Points must be 2D arrays of shape (2,)")
    
    if P1.shape != (3, 4) or P2.shape != (3, 4):
        raise ValueError("Projection matrices must be 3x4")
    
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


# Import constant for use in this module
try:
    from .constants import MAX_IMAGE_DIMENSION
except ImportError:
    MAX_IMAGE_DIMENSION = 100000  # Fallback

