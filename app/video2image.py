import cv2
import os
import argparse
from pathlib import Path

try:
    from PySide6.QtCore import QThread, Signal
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


def extract_frames(video_path, output_dir=None, downsample=30, progress_callback=None):
    """
    Extract frames from a video file and save them as JPG images.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str, optional): Directory to save images. If None, uses directory
                                    with the same name as the video (without extension)
        downsample (int): Downsampling ratio. Save every Nth frame (default: 30)
        progress_callback (callable, optional): Function to call with progress updates.
                                               Called as progress_callback(message, current, total)
    
    Returns:
        int: Number of frames extracted
    """
    # Convert to Path object for easier path manipulation
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory name based on video name (without extension)
    if output_dir is None:
        output_dir = video_path.parent / video_path.stem
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path.name}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Downsample ratio: {downsample} (saving every {downsample} frames)")
    print(f"Output directory: {output_dir}")
    
    if progress_callback:
        progress_callback(f"Extracting frames from video...", 0, total_frames)
    
    frame_count = 0
    saved_count = 0
    
    # Extract and save frames
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame only if it matches the downsampling ratio
        if frame_count % downsample == 0:
            frame_filename = output_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        
        # Progress indicator
        if progress_callback:
            # Update progress every frame for smoother progress bar
            progress_callback(
                f"Extracting frames... ({frame_count + 1}/{total_frames}, saved: {saved_count})",
                frame_count + 1,
                total_frames
            )
        elif (frame_count + 1) % 100 == 0:
            print(f"Processed {frame_count + 1}/{total_frames} frames...")
        
        frame_count += 1
    
    # Release video capture
    cap.release()
    
    if progress_callback:
        progress_callback(f"Extraction complete! Saved {saved_count} frames.", total_frames, total_frames)
    
    print(f"\nExtraction complete!")
    print(f"Saved {saved_count} frames to {output_dir}")
    
    return saved_count


def main():
    """Command-line interface for the video to image converter."""
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file and save them as JPG images.",
        epilog="Example: python video2image.py /path/to/video.mp4"
    )
    parser.add_argument(
        "video_path",
        type=str,
        metavar="PATH",
        help="Path to the input video file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for images (default: directory named after video)"
    )
    
    args = parser.parse_args()
    
    try:
        extract_frames(args.video_path, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if QT_AVAILABLE:
    class VideoExtractionThread(QThread):
        """Thread for extracting video frames with progress updates"""
        progress_signal = Signal(str, int, int)  # message, current, total (0, 0 for indeterminate)
        finished_signal = Signal(dict)  # {'success': bool, 'saved_count': int, 'output_dir': str, 'error': str}
        
        def __init__(self, video_path, output_dir=None, downsample=30):
            super().__init__()
            self.video_path = video_path
            self.output_dir = output_dir
            self.downsample = downsample
        
        def run(self):
            """Run video extraction in background thread"""
            def progress_callback(message, current, total):
                self.progress_signal.emit(message, current, total)
            
            try:
                saved_count = extract_frames(
                    self.video_path,
                    output_dir=self.output_dir,
                    downsample=self.downsample,
                    progress_callback=progress_callback
                )
                
                # Determine output directory
                video_path = Path(self.video_path)
                if self.output_dir is None:
                    output_dir = str(video_path.parent / video_path.stem)
                else:
                    output_dir = str(self.output_dir)
                
                self.finished_signal.emit({
                    'success': True,
                    'saved_count': saved_count,
                    'output_dir': output_dir
                })
            except Exception as e:
                self.finished_signal.emit({
                    'success': False,
                    'error': str(e),
                    'saved_count': 0,
                    'output_dir': None
                })


if __name__ == "__main__":
    exit(main())


