import cv2
import numpy as np


def dark_channel(image, window_size=15):
    """Calculate the dark channel of an image."""
    min_channel = np.min(image, axis=2)
    return cv2.erode(min_channel, np.ones((window_size, window_size)))


def estimate_atmosphere(image, dark_channel, percentile=0.001):
    """Estimate the atmosphere light of the image."""
    flat_dark_channel = dark_channel.flatten()
    flat_image = image.reshape(-1, 3)
    num_pixels = flat_image.shape[0]
    num_pixels_to_keep = int(num_pixels * percentile)
    indices = np.argpartition(flat_dark_channel, -num_pixels_to_keep)[-num_pixels_to_keep:]
    atmosphere = np.max(flat_image[indices], axis=0)
    return atmosphere


def dehaze(image, tmin=0.1, omega=0.95, window_size=15):
    """Dehaze the input image using the Dark Channel Prior algorithm."""
    if image is None:
        return None

    image = image.astype(np.float64) / 255.0
    dark_ch = dark_channel(image, window_size)
    atmosphere = estimate_atmosphere(image, dark_ch)
    transmission = 1 - omega * dark_ch
    transmission = np.maximum(transmission, tmin)
    dehazed = np.zeros_like(image)
    for channel in range(3):
        dehazed[:, :, channel] = (image[:, :, channel] - atmosphere[channel]) / transmission + atmosphere[channel]
    dehazed = np.clip(dehazed, 0, 1)
    dehazed = (dehazed * 255).astype(np.uint8)
    return dehazed


if __name__ == "__main__":
    # Specify the path to the video file
    video_path = r'C:\Users\hp\Downloads\input_video.mp4'

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the width and height of the video frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the dehazed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = r'C:\Users\hp\Downloads\output_video.mp4'
    output_video = cv2.VideoWriter(output_video_path, fourcc, 25.0, (frame_width, frame_height))

    while True:
        # Read the next frame from the video file
        ret, frame = cap.read()

        # If the frame is not empty, dehaze it and write it to the output video
        if ret:
            dehazed_frame = dehaze(frame)
            output_video.write(dehazed_frame)

            # Display the dehazed frame
            cv2.imshow('Dehazed Video', dehazed_frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # If the frame is empty, break the loop
        else:
            break

    # Release the video capture and video writer objects
