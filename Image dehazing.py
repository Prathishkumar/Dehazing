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
    # Specify the path to your image file
    image_path = image_path = r'C:\Users\hp\Desktop\input_image.jpg'


    # Attempt to load the image
    input_image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if input_image is not None:
        # Apply dehazing
        output_image = dehaze(input_image)

        if output_image is not None:
            # Save the dehazed image with the desired output path and filename
            cv2.imwrite('output.jpg', output_image)

            # Display the dehazed image
            cv2.imshow('Dehazed Image', output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Failed to dehaze the image.")
    else:
        # The image loading failed
        print("Error: Could not load the input image.")
