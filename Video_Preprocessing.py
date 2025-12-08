import cv2 
import numpy as np
import matplotlib.pyplot as plt
from moviepy import ImageSequenceClip # type: ignore
import os
import sys
TARGET_SIZE = (224, 224)
FRAME_NUMBER = 64
data = sys.argv[1] if len(sys.argv) > 1 else "Fall"
output_path = "Processed_Data/" + data
if not os.path.exists(output_path):
    os.makedirs(output_path)
for path in os.listdir(f'./Dataset/{data}/Raw_Video/'):
    input_path = os.path.join(f'./Dataset/{data}/Raw_Video/', path)
    cap = cv2.VideoCapture(input_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, TARGET_SIZE)
        frames.append(frame_resized)

    cap.release()

    # Adjust number of frames
    if len(frames) < FRAME_NUMBER:
        last_frame = frames[-1]
        while len(frames) < FRAME_NUMBER:
            frames.append(last_frame)
    else:
        # Uniformly sample frames
        step = len(frames) / FRAME_NUMBER
        frames = [frames[int(i * step)] for i in range(FRAME_NUMBER)]

    # Save processed video
    clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=15)
    output_file = os.path.join(output_path, path)
    clip.write_videofile(output_file, codec='libx264')