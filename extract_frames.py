"""
This file takes in a root directory containing video files and an output directory.
It extracts frames from each video file at the given fps and saves them to the output directory.
The folders in the output directory will be named after the video files.
"""


import os

import ffmpeg
import tqdm


def extract_frames(video_path: str, output_folder: str, fps: int) -> None:
    assert os.path.exists(video_path), "Video file does not exist"
    assert os.path.exists(output_folder), "Output folder does not exist"

    # Build the ffmpeg command to extract frames
    (
        ffmpeg
        .input(video_path)
        .output(os.path.join(output_folder, 'frame_%04d.jpg'), 
                start_number=0, 
                format='image2',
                vcodec='mjpeg',
                vf='fps=1')
        .run(capture_stdout=True, capture_stderr=True)
    )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', type=str, required=True, help='Path to directory with video files')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output directory')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second to extract')
    
    args = parser.parse_args()
    videos = os.listdir(args.videos_path)

    videos = [os.path.join(args.videos_path, video) for video in videos if video.endswith('.mp4')]
    videos = [video for video in videos if "DRONE" in video]
    assert os.path.exists(args.output_folder), "Output folder does not exist"
    
    for video in tqdm.tqdm(videos):
        print(video)
        output_frame_folder = os.path.join(args.output_folder, os.path.basename(video).split('.')[0])
        os.makedirs(output_frame_folder, exist_ok=True)
        extract_frames(video, output_frame_folder, args.fps)