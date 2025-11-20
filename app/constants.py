"""
Constants and configuration values for the application
"""

# Keypoint labeling constants
KEYPOINT_REMOVAL_THRESHOLD = 10.0  # pixels - distance threshold for keypoint removal
RECOMMENDATION_DISTANCE_THRESHOLD = 50.0  # pixels - threshold for ID recommendations
KEYPOINT_DISPLAY_SIZE = 5  # pixels - radius for keypoint display
KEYPOINT_LABEL_OFFSET_X = 8  # pixels - x offset for keypoint labels
KEYPOINT_LABEL_OFFSET_Y = -8  # pixels - y offset for keypoint labels

# ArUco marker visualization
MARKER_AXIS_LENGTH_RATIO = 0.3  # 30% of marker size for axis visualization
MARKER_CORNER_SIZE = 3  # pixels - size of corner markers
MARKER_BORDER_WIDTH = 2  # pixels - width of marker border

# Camera visualization
CAMERA_AXIS_LENGTH = 0.1  # meters - length of camera axis visualization
ARUCO_AXIS_LENGTH = 0.05  # meters - length of ArUco axis visualization

# UI Constants
DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 900
DEFAULT_MARKER_SIZE = 0.05  # meters

# File operations
DEFAULT_JPEG_QUALITY = 95
MAX_IMAGE_DIMENSION = 100000  # pixels - safety limit

# Triangulation
MIN_OBSERVATIONS_FOR_TRIANGULATION = 3
TRIANGULATION_CONSISTENCY_THRESHOLD = 0.1  # meters - max difference between triangulation pairs

# Bounding box calculation
BBOX_PADDING_RATIO = 0.01  # 1% padding for bounding boxes
MIN_BBOX_DIMENSION = 1  # pixels - minimum bbox width/height

# Visibility check
VISIBILITY_THRESHOLD = 85.0  # degrees - angle threshold for object visibility

