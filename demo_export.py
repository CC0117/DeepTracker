import cv2
import os
from tracker import Tracker


def run_demo(media_path, output_video_path):
    """ Main function to process images or video for object tracking. """
    # check if the media_path is a directory containing images
    if os.path.isdir(media_path):
        # create a list of .jpg files sorted by name
        media_files = sorted([os.path.join(media_path, f) for f in os.listdir(media_path) if f.endswith('.jpg')])
        frame_source = "image"
    # check if the media_path is a video file
    elif os.path.isfile(media_path) and media_path.endswith(('.mp4', '.avi', '.mov')):
        media_files = [media_path]
        frame_source = "video"
    else:
        print("Error: Invalid media path provided.")
        return

    # check if any media files were found
    if not media_files:
        print("Error: No media files found.")
        return

    print('Initiating tracker')
    tracker = Tracker()
    out = None
    max_frames = 600  # 30 seconds at 20 FPS
    frame_idx = 0

    print(f"Starting to process {len(media_files)} {'images' if frame_source == 'image' else 'video file(s)'}.")

    # process each media file
    for media_file in media_files:
        if frame_source == "video":
            print(f"Opening video file: {media_file}")
            cap = cv2.VideoCapture(media_file)
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video file reached.")
                    break
                # initialize video writer once with the first frame
                if out is None and frame is not None:
                    height, width = frame.shape[:2]
                    # initialize video writer with the same resolution as the input video
                    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
                    print(f"Video writer initialized with resolution: {width}x{height}")
                process_frame(frame, frame_idx, tracker, out, max_frames)
                frame_idx += 1
            cap.release()
        else:
            frame = cv2.imread(media_file)
            if frame is None:
                print(f"Error: Could not read image {media_file}.")
                continue
            # initialize video writer once with the first image
            if out is None:
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
                print(f"Video writer initialized with resolution: {width}x{height}")
            print(f"Processing image {media_file} at index {frame_idx}")
            process_frame(frame, frame_idx, tracker, out, max_frames)
            frame_idx += 1

    # release the video writer if it was used
    if out:
        out.release()
        print("Video writer released.")

    print("Video processing complete. Output saved to:", output_video_path)


def process_frame(frame, frame_idx, tracker, out, max_frames):
    """ Processes a single frame, updating it with the tracker and writing to the output video writer. """
    # check if the maximum frame count has been reached
    if frame_idx >= max_frames:
        print("Reached the 30-second limit.")
        return
    print(f"Processing frame {frame_idx}")
    # update the frame with the tracker
    frame, _ = tracker.update(frame, frame_idx)
    # write the updated frame to the video writer
    out.write(frame)


if __name__ == '__main__':
    media_path = 'street.mov'  # this can be a directory or a video file
    output_video_path = 'output.mp4'
    run_demo(media_path, output_video_path)
