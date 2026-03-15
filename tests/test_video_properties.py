"""
效果展示(初步)(清晰度、重叠点、平均移动距离)
Created by Gemini
"""
import sys
import os
import cv2
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.calculate.properties import image_clarity, overleap_point, ave_move

def test_video_properties():
    # video path
    video_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "examples", "video", "H02.mp4"
    )
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_distance = 1.0 / fps if fps > 0 else 0.0
    
    clarity_list = []
    overlap_list = []
    move_list = []
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read first frame")
        return
        
    clarity_list.append(image_clarity(prev_frame))
    overlap_list.append(0)  # No adjacent frame for the first one
    move_list.append(0.0)   # No adjacent frame for the first one
    
    frame_count = 1
    # Process up to a certain number of frames for demonstration purposes, e.g. 100 frames. 
    # Or, uncomment the limit to process the whole video.
    max_frames = 300
    
    print(f"Starting to process video. Total frames to process: {max_frames}")
    while True:
        ret, curr_frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
            
        c = image_clarity(curr_frame)
        o = overleap_point(prev_frame, curr_frame)
        m = ave_move(prev_frame, curr_frame)
        
        clarity_list.append(c)
        overlap_list.append(o)
        move_list.append(m)
        
        prev_frame = curr_frame
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}")
        
    cap.release()
    print("Video processing completed. Generating plots...")
    
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    
    frames = range(1, len(clarity_list) + 1)
    
    axs[0].plot(frames, clarity_list, label='Clarity (Variance of Laplacian)', color='blue', marker='o', markersize=3)
    
    # Calculate and plot the average clarity
    avg_clarity = sum(clarity_list) / len(clarity_list) if clarity_list else 0
    axs[0].axhline(y=avg_clarity, color='purple', linestyle='-', linewidth=2, label=f'Average Clarity: {avg_clarity:.2f}')
    axs[0].legend()
    
    axs[0].set_title('Image Clarity per Frame')
    axs[0].set_ylabel('Clarity')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Shift overlap and move distance so they represent the distance/overlap between current and previous frame
    # and dropping the 0 values of the first frame for the plot or keeping them
    axs[1].plot(frames, overlap_list, label='Overlap Points', color='green', marker='o', markersize=3)
    axs[1].set_title('Overlapping Feature Points with Previous Frame')
    axs[1].set_ylabel('Overlap Points')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    axs[2].plot(frames, move_list, label='Average Move Distance', color='red', marker='o', markersize=3)
    axs[2].set_title('Average Feature Move Distance with Previous Frame')
    axs[2].set_xlabel('Frame Index')
    axs[2].set_ylabel('Average Distance')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    # Add time_distance in the upper right
    plt.figtext(0.95, 0.95, f'time_distance = {time_distance:.4f} s', 
                fontsize=12, ha='right', va='top', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "test_result"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "video_properties_stats.png")
    plt.savefig(output_path, dpi=300)
    print(f"Test completed, statistics plot saved to {output_path}")

if __name__ == "__main__":
    test_video_properties()
