# Default camera intrinsic parameters
fx = 690.859619
fy = 691.077819
cx = 644.963989
cy = 360.913696

k1 = 0.0044435919262468815
k2 = -0.04403187334537506
k3 = 0.03050459735095501
k4 = 0.0
k5 = 0.0
k6 = 0.0
p1 = 0.00020420094369910657
p2 = 0.00019879947649315

# Default ArUco marker parameters
physical_length = 0.164  # meters
aruco_size = '4x4'  # Options: '4x4', '5x5', '6x6'
min_aruco_count = 3  # Minimum number of ArUco markers required per image (images with fewer markers will be ignored)

# Keypoint parameters
max_keypoint_num = 4  # Maximum number of keypoint IDs (keypoint IDs are in range [1, max_keypoint_num])
max_cls_id = 3  # Maximum class ID (class IDs are in range [1, max_cls_id])

recommend_image_num = 3000

# Default video extraction parameters
downsample_ratio = 15  # Save every Nth frame when extracting frames from video

# Bundle adjustment option
enable_ba = True  # Enable bundle adjustment (default: False)

# False positive detection parameters
isolation_neighboor_num = 2  # Number of neighbor frames to check for false positive ArUco markers (markers not present in any neighbor frames are considered false positives)

# ArUco detector parameters (for suppressing false positives)
aruco_min_corner_distance_rate = 0.1  # Minimum distance between corners (relative to marker perimeter)
aruco_min_marker_distance_rate = 0.1  # Minimum distance between markers (relative to marker perimeter)
aruco_polygonal_approx_accuracy_rate = 0.02  # Accuracy rate for polygonal approximation (more strict than default 0.03)
aruco_min_marker_perimeter_rate = 0.05  # Minimum marker perimeter rate (filters too small markers)
aruco_max_marker_perimeter_rate = 2.0  # Maximum marker perimeter rate (filters too large markers)
aruco_min_otsu_std_dev = 7.0  # Minimum Otsu threshold standard deviation (filters flat histograms)
aruco_adaptive_thresh_constant = 5  # Adaptive threshold constant (smaller = less expansion)
aruco_error_correction_rate = 0.3  # Error correction rate (lower = more conservative, default 0.6)

# Visibility check parameters
visibility_threshold = 85.0  # degrees - angle threshold for object visibility (default: 85 degrees)

# PyTorch bundle adjustment parameters
ba_learning_rate = 1e-3  # Learning rate for PyTorch bundle adjustment optimization
ba_max_iterations = 1000  # Maximum number of iterations for PyTorch bundle adjustment

# Reprojection error filtering parameters
reprojection_error_threshold = 10.0  # Maximum average reprojection error per frame (in pixels). Frames with higher error will be filtered out.