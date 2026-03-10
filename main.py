from utils import histogram, i_frames, blur, edge_detect, encoder, decoder

import numpy as np
import torch
import cv2


VIDEOFILE = "test_videos/bowing_cif.avi"
MEMBUDGET = 7_000_000_000 # 7GB
DEVICE    = "cuda:0"

cap = cv2.VideoCapture(VIDEOFILE)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()


video_rgb = []

while True:
    ret, frame = cap.read()  # ret is a boolean (True if frame is read correctly), frame is the image array in BGR
    
    # If the frame was not read correctly (e.g., end of video), break the loop
    if not ret:
        break
    
    frame = frame[:, :, ::-1] # convert from BGR to RGB
    video_rgb.append(frame)

cap.release()

video_rgb = np.asarray(video_rgb)
video_rgb = torch.as_tensor(video_rgb, dtype=torch.float32, device=DEVICE)

rgb_to_grayscale_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=DEVICE)
rgb_to_grayscale_weights = rgb_to_grayscale_weights.view(1,1,1,3)

weighted_video  = video_rgb * rgb_to_grayscale_weights
video_grayscale = torch.sum(weighted_video, dim=-1)

blurred_grayscale = blur.blur_video(video_grayscale)
blurred_histogram = histogram.video_histogram(blurred_grayscale, MEMBUDGET)

video_edges       = edge_detect.sobel(blurred_grayscale)

troi_slices       = histogram.temporal_region_of_interest(blurred_histogram)
i_frame_indices   = i_frames.select_i_frames(video_edges, troi_slices)

encoded_video = encoder.encode_video(video_rgb, i_frame_indices)

# Writing encoded video
with open("compressed.ztensor", "wb") as f:
   f.write(encoded_video)



decoded_video = decoder.decode(encoded_video)

# Writing decoded video
decoded_video = decoded_video.cpu().numpy().astype(np.uint8)[:, :, :, ::-1]

with open("decompressed.raw", "wb") as f:
    f.write(decoded_video.tobytes())
