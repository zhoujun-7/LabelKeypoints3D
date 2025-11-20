"""
Convert JSON labeling results to YOLO format labels

Each image gets a corresponding .txt file with the same name.
Format: <class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> ...
"""

import json
import os
import sys
import cv2
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List, Dict, Any

# Try to import utility functions, fallback if not available
try:
    from app.utils import safe_read_json, safe_join_path, calculate_bbox_safe, validate_image_dimensions
    USE_UTILS = True
except ImportError:
    USE_UTILS = False
    # Fallback implementations
    def safe_read_json(json_path: str) -> Dict[str, Any]:
        """Fallback JSON reading"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def safe_join_path(base_dir: str, filename: str) -> Path:
        """Fallback path joining"""
        return Path(base_dir) / filename
    
    def calculate_bbox_safe(keypoints_2d: List[List[float]]) -> Optional[Tuple[float, float, float, float]]:
        """Fallback bbox calculation"""
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
    
    def validate_image_dimensions(img) -> bool:
        """Fallback image validation"""
        return img is not None


def load_image_dimensions(image_path: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Load image to get dimensions with error handling.
    
    Args:
        image_path: Path to image file
    
    Returns:
        (width, height) or (None, None) if failed
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None
        
        if not validate_image_dimensions(img):
            return None, None
        
        h, w = img.shape[:2]
        return w, h
    except Exception as e:
        print(f"Warning: Error loading image {image_path}: {e}")
        return None, None


def normalize_coordinates(x: float, y: float, img_width: int, img_height: int) -> Tuple[float, float]:
    """
    Normalize coordinates to [0, 1] range with validation.
    
    Args:
        x: X coordinate
        y: Y coordinate
        img_width: Image width
        img_height: Image height
    
    Returns:
        Normalized (x, y) coordinates
    
    Raises:
        ValueError: If image dimensions are invalid
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")
    return x / img_width, y / img_height


def convert_json_to_yolo(json_path: str, image_dir: Optional[str] = None, output_dir: Optional[str] = None) -> str:
    """
    Convert JSON labeling results to YOLO format with improved error handling.
    
    Args:
        json_path: Path to the JSON file
        image_dir: Directory containing images (if None, inferred from JSON path)
        output_dir: Output directory for labels (if None, uses json_dir + '_label')
    
    Returns:
        Path to output directory
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON format is invalid
        PermissionError: If cannot write to output directory
    """
    json_path = Path(json_path)
    
    # Validate JSON file exists
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Load JSON file with error handling
    try:
        data = safe_read_json(str(json_path))
    except (FileNotFoundError, PermissionError, ValueError) as e:
        raise
    
    # Extract keypoints
    if isinstance(data, dict) and 'keypoints' in data:
        keypoints = data['keypoints']
    elif isinstance(data, list):
        keypoints = data
    else:
        raise ValueError(f"Invalid JSON format: expected dict with 'keypoints' key or list, got {type(data)}")
    
    # Get JSON directory
    json_dir = json_path.parent.resolve()
    
    # Determine image directory
    if image_dir is None:
        # Try to infer from JSON path or use JSON directory
        # Check if there's a corresponding image directory
        json_basename = json_path.name
        if json_basename == 'labeling_results.json':
            # Check for _valid directory
            if json_dir.name.endswith('_valid'):
                # Use parent directory
                image_dir = str(json_dir.parent)
            else:
                image_dir = str(json_dir)
        else:
            image_dir = str(json_dir)
    else:
        image_dir = str(Path(image_dir).resolve())
    
    # Validate image directory exists
    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory does not exist: {image_dir}")
    
    # Create output directory
    if output_dir is None:
        output_dir = str(json_dir / f"{json_dir.name}_label")
    else:
        output_dir = str(Path(output_dir).resolve())
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Permission denied creating output directory: {output_dir}")
    except OSError as e:
        raise OSError(f"Failed to create output directory {output_dir}: {e}")
    
    print(f"Output directory: {output_dir}")
    
    # Group keypoints by image_name
    keypoints_by_image = defaultdict(lambda: defaultdict(list))
    
    for kp in keypoints:
        image_name = kp.get('image_name', '')
        object_id = kp.get('object_id', 1)
        keypoint_id = kp.get('keypoint_id', 1)
        keypoint_2d = kp.get('keypoint_2d', None)
        cls_id = kp.get('cls_id', 1)
        
        if keypoint_2d is None:
            continue
        
        # Store keypoint with its ID, class, and visibility
        visibility = kp.get('visibility', True)  # Default to True if not present (backward compatibility)
        # Convert to int: True -> 1 (visible), False -> 0 (invisible)
        if isinstance(visibility, bool):
            visibility_int = 1 if visibility else 0
        elif isinstance(visibility, (int, float)):
            # If it's already a number, use it directly (clamp to valid range 0-1)
            visibility_int = int(max(0, min(1, visibility)))
        else:
            # Default to visible if invalid type
            visibility_int = 1
        
        keypoints_by_image[image_name][object_id].append({
            'keypoint_id': keypoint_id,
            'keypoint_2d': keypoint_2d,
            'cls_id': cls_id,
            'visibility': visibility_int
        })
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    
    for image_name, objects in keypoints_by_image.items():
        # Find image file with path validation
        image_path = None
        try:
            # Validate image name doesn't contain path traversal
            safe_image_name = safe_join_path(image_dir, image_name)
            if safe_image_name.exists() and safe_image_name.is_file():
                image_path = str(safe_image_name)
        except ValueError:
            # Path traversal detected, try to find by name only
            pass
        
        # If not found, try common extensions
        if image_path is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            base_name = Path(image_name).stem  # Remove extension if present
            
            for ext in image_extensions:
                try:
                    test_path = safe_join_path(image_dir, base_name + ext)
                    if test_path.exists() and test_path.is_file():
                        image_path = str(test_path)
                        break
                except ValueError:
                    continue
        
        if image_path is None:
            print(f"Warning: Image not found for {image_name}, skipping...")
            skipped_count += 1
            continue
        
        # Load image dimensions
        img_width, img_height = load_image_dimensions(image_path)
        if img_width is None or img_height is None:
            print(f"Warning: Could not load image {image_path}, skipping...")
            skipped_count += 1
            continue
        
        # Create label file
        label_filename = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        # Process each object in the image
        label_lines = []
        
        for object_id, keypoints_list in objects.items():
            # Sort keypoints by keypoint_id
            keypoints_list.sort(key=lambda x: x['keypoint_id'])
            
            # Get class ID (use the first keypoint's cls_id for this object)
            cls_id = keypoints_list[0]['cls_id']
            class_index = cls_id - 1  # YOLO uses 0-indexed classes
            
            # Extract 2D coordinates and visibility with validation
            keypoints_2d = []
            keypoints_visibility = []
            for kp in keypoints_list:
                kp_2d = kp.get('keypoint_2d')
                if kp_2d and isinstance(kp_2d, (list, tuple)) and len(kp_2d) >= 2:
                    try:
                        keypoints_2d.append([float(kp_2d[0]), float(kp_2d[1])])
                        # Get visibility value (already converted to int in first loop, default to 1)
                        visibility = kp.get('visibility', 1)
                        keypoints_visibility.append(visibility)
                    except (ValueError, TypeError):
                        continue
            
            if not keypoints_2d:
                continue
            
            # Calculate bounding box with safe function
            bbox = calculate_bbox_safe(keypoints_2d)
            if bbox is None:
                continue
            
            x_min, y_min, x_max, y_max = bbox
            
            # Calculate normalized bbox center and size
            bbox_center_x = (x_min + x_max) / 2.0
            bbox_center_y = (y_min + y_max) / 2.0
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            # Normalize bbox coordinates
            norm_center_x, norm_center_y = normalize_coordinates(
                bbox_center_x, bbox_center_y, img_width, img_height
            )
            norm_width = bbox_width / img_width
            norm_height = bbox_height / img_height
            
            # Normalize keypoint coordinates with validation
            normalized_keypoints = []
            for idx, kp_2d in enumerate(keypoints_2d):
                try:
                    norm_x, norm_y = normalize_coordinates(
                        kp_2d[0], kp_2d[1], img_width, img_height
                    )
                    # Clamp to [0, 1] range
                    norm_x = max(0.0, min(1.0, norm_x))
                    norm_y = max(0.0, min(1.0, norm_y))
                    normalized_keypoints.append(norm_x)
                    normalized_keypoints.append(norm_y)
                    # Use visibility from JSON (default to 1 if not available)
                    visibility = keypoints_visibility[idx] if idx < len(keypoints_visibility) else 1
                    normalized_keypoints.append(visibility)
                except (ValueError, TypeError, ZeroDivisionError):
                    # Skip invalid keypoint
                    continue
            
            # Format: <class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> ...
            line_parts = [
                str(class_index),
                f"{norm_center_x:.6f}",
                f"{norm_center_y:.6f}",
                f"{norm_width:.6f}",
                f"{norm_height:.6f}"
            ]
            
            # Add keypoints
            for i in range(0, len(normalized_keypoints), 3):
                line_parts.append(f"{normalized_keypoints[i]:.6f}")
                line_parts.append(f"{normalized_keypoints[i+1]:.6f}")
                line_parts.append(f"{normalized_keypoints[i+2]:.0f}")
            
            label_lines.append(" ".join(line_parts))
        
        # Write label file with error handling
        if label_lines:
            try:
                label_path_obj = Path(label_path)
                label_path_obj.parent.mkdir(parents=True, exist_ok=True)
                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(label_lines))
                processed_count += 1
                print(f"Created label file: {label_path} ({len(label_lines)} objects)")
            except (PermissionError, OSError) as e:
                print(f"Error: Could not write label file {label_path}: {e}")
                skipped_count += 1
        else:
            print(f"Warning: No valid keypoints for {image_name}, skipping label file...")
            skipped_count += 1
    
    print(f"\nConversion complete!")
    print(f"  Processed: {processed_count} images")
    print(f"  Skipped: {skipped_count} images")
    print(f"  Output directory: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_to_YoloLabel.py <json_file> [image_dir] [output_dir]")
        print("\nExample:")
        print("  python json_to_YoloLabel.py labeling_results.json")
        print("  python json_to_YoloLabel.py labeling_results.json /path/to/images")
        print("  python json_to_YoloLabel.py labeling_results.json /path/to/images /path/to/output")
        sys.exit(1)
    
    json_path = sys.argv[1]
    image_dir = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        result = convert_json_to_yolo(json_path, image_dir, output_dir)
        print(f"\nSuccess! Output saved to: {result}")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

