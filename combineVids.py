import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
import pandas as pd


mouseVid = '/Volumes/Data/BMAD/TRAP 2024/TRAP_Round3_iso/05_15_2024_Gq_CNO/05_15_2024_Gq_videos/708_MA149_CNO1_0515.mp4'
vid1 = cv2.VideoCapture(mouseVid)

behaviorFilePath = '/Volumes/Data/BMAD/BMAD Team/allAnimalsAllBehaviors_featuresAdded_trainTestSplit_manualAnnotations_predictionsAdded_thresholded.csv'
behaviorFile = pd.read_csv(behaviorFilePath)
behavior_data = behaviorFile[behaviorFile['BMAD Filename'] == 'Z:/BMAD/TRAP 2024/TRAP_Round3_iso/05_15_2024_Gq_CNO/708_MA149/20240515_T110923-07_MA149_0.csv']
behavior_data = behavior_data[:18000]
print("Finished reading file")
offset = int(behavior_data.iloc[0]['Frame Offset'])
print(f"offset: {str(offset)}")

start_frame = offset * 30

animal_id = "708_MA149"
trial = "C1"

animatedVid = '/Volumes/Data/BMAD/BMAD Team/combine videos/animated videos/708_MA149_C1_4x.mp4'
vid2 = cv2.VideoCapture(animatedVid)

behaviorw = int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH))
behaviorh = int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps_vid1 = int(vid1.get(cv2.CAP_PROP_FPS))
fps_vid2 = int(vid2.get(cv2.CAP_PROP_FPS))

# final output is at vid2's FPS
final_fps = fps_vid2

animw = int(vid2.get(cv2.CAP_PROP_FRAME_WIDTH))
animh = int(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT))

final_width = max(behaviorw, animw)
final_height = behaviorh + animh

out = cv2.VideoWriter(
    f'/Volumes/Data/BMAD/BMAD Team/combine videos/combined videos/{animal_id}_{trial}_4x.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    final_fps,
    (final_width, final_height)
)

vid1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

jumps = 0

csv_index = 0
csv_data = behavior_data.to_dict(orient='records')

while True:
    # read every 4th frame from vid1
    for _ in range(4):
        success_vid1, behavior_frame = vid1.read()
        if not success_vid1:
            break

    # read every frame from vid2
    success_vid2, anim_frame = vid2.read()

    if not success_vid1 or not success_vid2 or csv_index >= len(csv_data):
        break

    if csv_index < len(csv_data):

        row = csv_data[csv_index]

        timestamp = row.get('Time', 'N/A')

        timestamp = f"{float(timestamp):.1f}" if timestamp != 'N/A' else "N/A"

        behaviors = []
        predicted_behaviors = []

        for behavior in ['rearing', 'grooming', 'circling', 'lay_on_belly', 'straub_tail']:
            value = row.get(behavior, 0)

            if pd.notna(value) and value not in [0, 'N/A', ""]:
                behaviors.append(behavior)

        for predicted in ['thresholded rearing', 'thresholded grooming', 'thresholded circling', 'thresholded lay_on_belly', 'thresholded straub_tail']:
            value = row.get(predicted, 0)

            if pd.notna(value) and value not in [0, 'N/A', ""]:
                predicted_behaviors.append(predicted)

        behavior_text = f"Behavior: {', '.join(behaviors) if behaviors else ''}"
        predicted_behavior_text = f"Predicted Behavior: {', '.join(predicted_behaviors) if predicted_behaviors else ''}"

    else:
        frame_number = "N/A"
        timestamp = "N/A"
        behavior_text = ""
        predicted_behavior_text = ""

    # combine the two frames
    final_frame = np.zeros((final_height, final_width, 3), dtype=np.uint8)

    if animw < final_width:
        pad_left = (final_width - animw) // 2
        pad_right = final_width - animw - pad_left
        anim_frame = cv2.copyMakeBorder(
            anim_frame, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    final_frame[0:behaviorh, :, :] = behavior_frame

    final_frame[behaviorh:behaviorh + animh, :, :] = anim_frame

    # add text
    text_color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.7
    thickness = 2
    cv2.putText(final_frame, f"Animal ID: {animal_id}", (20, 60), font, font_scale, text_color, thickness)
    cv2.putText(final_frame, f"Timestamp: {timestamp}", (20, 120), font, font_scale, text_color, thickness)
    cv2.putText(final_frame, f"Frame #: {csv_index}", (20, 180), font, font_scale, text_color, thickness)
    cv2.putText(final_frame, behavior_text, (20, 240), font, font_scale, text_color, thickness)
    cv2.putText(final_frame, predicted_behavior_text, (20, 300), font, font_scale, text_color, thickness)

    out.write(final_frame)

    csv_index += 4

    if csv_index % final_fps == 0:
        print(f"Processed {csv_index // final_fps} seconds of video")

vid1.release()
vid2.release()
out.release()

print("Video processing complete")