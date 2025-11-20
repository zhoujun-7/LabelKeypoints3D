"""
Trajectory visualization tab for ArUco poses and camera trajectory
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressDialog
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QApplication
import numpy as np
import cv2
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Qt5Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[TRAJECTORY] Warning: matplotlib not available, trajectory visualization disabled")


class TrajectoryTab(QWidget):
    def __init__(self):
        super().__init__()
        self.aruco_poses = {}
        self.camera_poses = {}
        self.camera_intrinsic = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib figure
            self.fig = Figure(figsize=(10, 8), facecolor='#1e1e1e')
            self.canvas = FigureCanvas(self.fig)
            self.canvas.setStyleSheet("background-color: #1e1e1e;")
            layout.addWidget(self.canvas)
        else:
            # Fallback if matplotlib not available
            no_matplotlib_label = QLabel("Matplotlib not available. Please install matplotlib to view trajectory.")
            no_matplotlib_label.setStyleSheet("color: #cccccc; padding: 20px;")
            layout.addWidget(no_matplotlib_label)
        
        # Status label
        self.status_label = QLabel("No data available. Click 'Start Processing' to visualize trajectory.")
        self.status_label.setStyleSheet("color: #cccccc; padding: 10px;")
        layout.addWidget(self.status_label)
    
    def update_visualization(self, aruco_poses, camera_poses, camera_intrinsic):
        """Update trajectory visualization with new data"""
        if not MATPLOTLIB_AVAILABLE:
            self.status_label.setText("Matplotlib not available. Cannot visualize trajectory.")
            return
        
        self.aruco_poses = aruco_poses
        self.camera_poses = camera_poses
        self.camera_intrinsic = camera_intrinsic
        
        print(f"[TRAJECTORY] Updating visualization with {len(aruco_poses)} ArUco poses and {len(camera_poses)} camera poses")
        
        if not aruco_poses and not camera_poses:
            self.status_label.setText("No trajectory data available.")
            return
        
        # Show progress dialog for visualization
        total_steps = len(aruco_poses) + len(camera_poses) + 5  # +5 for setup steps
        progress_dialog = QProgressDialog("Visualizing trajectory...", None, 0, total_steps, self)
        progress_dialog.setWindowTitle("Visualization")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setCancelButton(None)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        progress_dialog.show()
        QApplication.processEvents()
        
        current_step = 0
        
        # Clear previous plots
        self.fig.clear()
        current_step += 1
        progress_dialog.setValue(current_step)
        progress_dialog.setLabelText("Clearing previous plots...")
        QApplication.processEvents()
        
        # Create 3D subplot
        ax = self.fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#1e1e1e')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#3c3c3c')
        ax.yaxis.pane.set_edgecolor('#3c3c3c')
        ax.zaxis.pane.set_edgecolor('#3c3c3c')
        ax.xaxis.label.set_color('#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        ax.zaxis.label.set_color('#cccccc')
        ax.tick_params(colors='#cccccc')
        ax.grid(True, color='#3c3c3c', alpha=0.3)
        
        # Initialize variables
        marker_positions = None
        camera_positions = None
        
        # Axis length for visualization (in meters)
        axis_length = 0.05  # 5cm for ArUco markers
        camera_axis_length = 0.1  # 10cm for camera
        
        # ArUco marker colors (dark colors)
        aruco_colors = {
            'x': '#8B0000',  # Dark red
            'y': '#006400',  # Dark green
            'z': '#00008B'   # Dark blue
        }
        
        # Camera colors (bright/highlight colors)
        camera_colors = {
            'x': '#FF0000',  # Bright red
            'y': '#00FF00',  # Bright green
            'z': '#0000FF'   # Bright blue
        }
        
        # Plot ArUco marker poses as RGB coordinate axes
        if aruco_poses:
            marker_pos_list = []
            marker_ids = []
            total_markers = len(aruco_poses)
            for idx, (marker_id, pose) in enumerate(aruco_poses.items()):
                tvec = np.array(pose['tvec'])
                rvec = np.array(pose['rvec'])
                marker_pos_list.append(tvec)
                marker_ids.append(marker_id)
                
                # Get rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                
                # Calculate axis directions (R columns are the axes in world frame)
                x_axis = R[:, 0] * axis_length
                y_axis = R[:, 1] * axis_length
                z_axis = R[:, 2] * axis_length
                
                # Plot X axis (red)
                ax.plot([tvec[0], tvec[0] + x_axis[0]], 
                       [tvec[1], tvec[1] + x_axis[1]], 
                       [tvec[2], tvec[2] + x_axis[2]], 
                       color=aruco_colors['x'], linewidth=2, alpha=0.8)
                
                # Plot Y axis (green)
                ax.plot([tvec[0], tvec[0] + y_axis[0]], 
                       [tvec[1], tvec[1] + y_axis[1]], 
                       [tvec[2], tvec[2] + y_axis[2]], 
                       color=aruco_colors['y'], linewidth=2, alpha=0.8)
                
                # Plot Z axis (blue)
                ax.plot([tvec[0], tvec[0] + z_axis[0]], 
                       [tvec[1], tvec[1] + z_axis[1]], 
                       [tvec[2], tvec[2] + z_axis[2]], 
                       color=aruco_colors['z'], linewidth=2, alpha=0.8)
                
                # Add label for marker
                ax.text(tvec[0], tvec[1], tvec[2], f'  ArUco {marker_id}', 
                       color='#90EE90', fontsize=9, fontweight='bold')
                
                # Update progress every 5 markers or at the end
                if (idx + 1) % 5 == 0 or (idx + 1) == total_markers:
                    current_step += 1
                    progress_dialog.setValue(current_step)
                    progress_dialog.setLabelText(f"Plotting ArUco markers... ({idx + 1}/{total_markers})")
                    QApplication.processEvents()
            
            if marker_pos_list:
                marker_positions = np.array(marker_pos_list)
        
        # Plot camera trajectory line first (before poses to ensure visibility)
        if camera_poses:
            current_step += 1
            progress_dialog.setValue(current_step)
            progress_dialog.setLabelText("Computing camera trajectory...")
            QApplication.processEvents()
            
            camera_pos_list = []
            frame_ids = []
            total_cameras = len(camera_poses)
            for idx, frame_id in enumerate(sorted(camera_poses.keys())):
                pose = camera_poses[frame_id]
                rvec = np.array(pose['rvec'])
                tvec = np.array(pose['tvec'])
                
                # Convert camera pose to world position
                R, _ = cv2.Rodrigues(rvec)
                # Camera position in world frame: -R^T @ t
                cam_pos = -R.T @ tvec
                camera_pos_list.append(cam_pos)
                frame_ids.append(frame_id)
                
                # Update progress every 20 frames or at the end
                if (idx + 1) % 20 == 0 or (idx + 1) == total_cameras:
                    progress_dialog.setLabelText(f"Computing camera trajectory... ({idx + 1}/{total_cameras})")
                    QApplication.processEvents()
            
            if camera_pos_list:
                camera_positions = np.array(camera_pos_list)
                print(f"[TRAJECTORY] Plotting camera trajectory with {len(camera_positions)} points")
                print(f"[TRAJECTORY] First point: {camera_positions[0]}, Last point: {camera_positions[-1]}")
                
                current_step += 1
                progress_dialog.setValue(current_step)
                progress_dialog.setLabelText("Plotting camera trajectory line...")
                QApplication.processEvents()
                
                # Plot camera trajectory line (red, thick, high z-order)
                ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                       color='#FF0000', linewidth=3, alpha=1.0, label='Camera Trajectory', zorder=1)
        
        # Plot camera poses as RGB coordinate axes (sample every few frames to avoid clutter)
        if camera_poses:
            current_step += 1
            progress_dialog.setValue(current_step)
            progress_dialog.setLabelText("Plotting camera poses...")
            QApplication.processEvents()
            
            sample_rate = max(1, len(camera_poses) // 20)  # Show ~20 camera poses
            sampled_frames = [fid for idx, fid in enumerate(sorted(camera_poses.keys())) 
                            if idx % sample_rate == 0 or idx == len(camera_poses) - 1]
            total_sampled = len(sampled_frames)
            
            for plot_idx, frame_id in enumerate(sampled_frames):
                idx = sorted(camera_poses.keys()).index(frame_id)
                pose = camera_poses[frame_id]
                rvec = np.array(pose['rvec'])
                tvec = np.array(pose['tvec'])
                
                # Convert camera pose to world position
                R, _ = cv2.Rodrigues(rvec)
                cam_pos = -R.T @ tvec
                
                # Calculate axis directions (R^T columns are the camera axes in world frame)
                # Camera X axis (right) in world frame
                x_axis = R.T[:, 0] * camera_axis_length
                # Camera Y axis (down) in world frame
                y_axis = R.T[:, 1] * camera_axis_length
                # Camera Z axis (forward) in world frame
                z_axis = R.T[:, 2] * camera_axis_length
                
                # Plot X axis (red) - lower z-order so trajectory is visible
                ax.plot([cam_pos[0], cam_pos[0] + x_axis[0]], 
                       [cam_pos[1], cam_pos[1] + x_axis[1]], 
                       [cam_pos[2], cam_pos[2] + x_axis[2]], 
                       color=camera_colors['x'], linewidth=2.5, alpha=0.9, zorder=2)
                
                # Plot Y axis (green) - lower z-order so trajectory is visible
                ax.plot([cam_pos[0], cam_pos[0] + y_axis[0]], 
                       [cam_pos[1], cam_pos[1] + y_axis[1]], 
                       [cam_pos[2], cam_pos[2] + y_axis[2]], 
                       color=camera_colors['y'], linewidth=2.5, alpha=0.9, zorder=2)
                
                # Plot Z axis (blue) - lower z-order so trajectory is visible
                ax.plot([cam_pos[0], cam_pos[0] + z_axis[0]], 
                       [cam_pos[1], cam_pos[1] + z_axis[1]], 
                       [cam_pos[2], cam_pos[2] + z_axis[2]], 
                       color=camera_colors['z'], linewidth=2.5, alpha=0.9, zorder=2)
                
                # Add label for camera (only for first and last)
                if idx == 0:
                    ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f'  Camera Start', 
                           color='#FFFF00', fontsize=9, fontweight='bold')
                elif idx == len(camera_poses) - 1:
                    ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f'  Camera End', 
                           color='#FFFF00', fontsize=9, fontweight='bold')
                
                # Update progress
                if (plot_idx + 1) % 5 == 0 or (plot_idx + 1) == total_sampled:
                    progress_dialog.setLabelText(f"Plotting camera poses... ({plot_idx + 1}/{total_sampled})")
                    QApplication.processEvents()
        
        # Set labels and title
        current_step += 1
        progress_dialog.setValue(current_step)
        progress_dialog.setLabelText("Setting up axes and labels...")
        QApplication.processEvents()
        
        ax.set_xlabel('X (m)', color='#cccccc')
        ax.set_ylabel('Y (m)', color='#cccccc')
        ax.set_zlabel('Z (m)', color='#cccccc')
        ax.set_title('ArUco Poses and Camera Trajectory\n(ArUco: Dark RGB axes, Camera: Bright RGB axes)', 
                    color='#cccccc', pad=20)
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#FF0000', lw=2, alpha=0.8, label='Camera Trajectory'),
            Line2D([0], [0], color=aruco_colors['x'], lw=2, label='ArUco X (Dark Red)'),
            Line2D([0], [0], color=aruco_colors['y'], lw=2, label='ArUco Y (Dark Green)'),
            Line2D([0], [0], color=aruco_colors['z'], lw=2, label='ArUco Z (Dark Blue)'),
            Line2D([0], [0], color=camera_colors['x'], lw=2.5, label='Camera X (Bright Red)'),
            Line2D([0], [0], color=camera_colors['y'], lw=2.5, label='Camera Y (Bright Green)'),
            Line2D([0], [0], color=camera_colors['z'], lw=2.5, label='Camera Z (Bright Blue)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 facecolor='#2d2d2d', edgecolor='#3c3c3c', labelcolor='#cccccc', fontsize=8)
        
        # Set equal aspect ratio
        all_points_list = []
        if camera_positions is not None and len(camera_positions) > 0:
            all_points_list.append(camera_positions)
        if marker_positions is not None and len(marker_positions) > 0:
            all_points_list.append(marker_positions)
        
        if all_points_list:
            all_points = np.vstack(all_points_list)
            if len(all_points) > 0:
                # Calculate bounds
                x_range = all_points[:, 0].max() - all_points[:, 0].min()
                y_range = all_points[:, 1].max() - all_points[:, 1].min()
                z_range = all_points[:, 2].max() - all_points[:, 2].min()
                max_range = max(x_range, y_range, z_range)
                
                if max_range > 0:
                    center_x = (all_points[:, 0].max() + all_points[:, 0].min()) / 2
                    center_y = (all_points[:, 1].max() + all_points[:, 1].min()) / 2
                    center_z = (all_points[:, 2].max() + all_points[:, 2].min()) / 2
                    
                    ax.set_xlim(center_x - max_range/2, center_x + max_range/2)
                    ax.set_ylim(center_y - max_range/2, center_y + max_range/2)
                    ax.set_zlim(center_z - max_range/2, center_z + max_range/2)

        
        # Update status
        status_text = f"Visualizing: {len(aruco_poses)} ArUco markers, {len(camera_poses)} camera poses"
        if camera_intrinsic is not None:
            status_text += f"\nCamera Intrinsic: fx={camera_intrinsic[0,0]:.2f}, fy={camera_intrinsic[1,1]:.2f}, cx={camera_intrinsic[0,2]:.2f}, cy={camera_intrinsic[1,2]:.2f}"
        self.status_label.setText(status_text)
        
        # Refresh canvas
        current_step += 1
        progress_dialog.setValue(current_step)
        progress_dialog.setLabelText("Rendering visualization...")
        QApplication.processEvents()
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Close progress dialog
        progress_dialog.close()
        
        print(f"[TRAJECTORY] Visualization updated successfully")

