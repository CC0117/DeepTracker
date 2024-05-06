import streamlit as st
import cv2
import tempfile
from PIL import Image
import os
import pandas as pd
from tracker import Tracker


def main():
    st.title("Video Tracking System")
    st.write("A tracking system based on YOLOv5m and DeepSORT.")

    # sidebar for user inputs
    st.sidebar.header("User Inputs")
    uploaded_file_live = st.sidebar.file_uploader("Upload Video for Live Tracking", type=["mp4", "avi", "mov"])

    # display gallery from a specified folder
    gallery_folder = 'Gallery'
    display_video_gallery(gallery_folder)

    # initialize the tracker
    tracker = Tracker()
    frame_idx = 0

    if uploaded_file_live is not None:
        # save the uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file_live.read())
        video_path = tfile.name

        # open the video file
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        # create containers for analytics
        num_people = st.empty()
        crowd_density = st.empty()
        track_duration_table = st.empty()

        # process video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # update the tracker with the current frame and get the annotated frame and analytics data
            processed_frame, analytics_data = tracker.update(frame, frame_idx)

            # update real-time analytics in the Streamlit app
            num_people.metric(label="Number of People Tracked", value=len(analytics_data['track_durations']))
            crowd_density.metric(label="Crowd Density", value=f"{analytics_data['crowd_density']:.2f}%")
            track_duration_table.dataframe(create_duration_table(analytics_data['track_durations']))

            # convert to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # display the processed frame in the main area
            stframe.image(pil_img, caption="Live Tracking", use_column_width=True)

            frame_idx += 1

        cap.release()
        os.unlink(video_path)


def create_duration_table(track_durations):
    """Create a DataFrame from track durations for display."""
    # convert the track durations dictionary into a DataFrame for nicer formatting
    df = pd.DataFrame(list(track_durations.items()), columns=['Track ID', 'Duration (frames)'])
    return df


def display_video_gallery(folder):
    """Function to display a video gallery with a select box."""
    st.write("## Video Gallery")

    # list video files in the specified folder
    video_files = sorted([f for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mov'))])

    if not video_files:
        st.write("No video files found in the directory.")
        return

    # create a select box to choose a video
    selected_video = st.selectbox("#### Choose a video", video_files)

    # display the selected video
    video_path = os.path.join(folder, selected_video)
    st.video(video_path)


if __name__ == '__main__':
    main()
