import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import matplotlib.gridspec as gridspec

# print("Choose behavior file:")
# behaviorFile = filedialog.askopenfilename()
# if behaviorFile:
#     print(f"Video: {behaviorFile}")
# else:
#     print("No behavior file selected")

behaviorFilePath = '/Volumes/Data/BMAD/BMAD Team/allAnimalsAllBehaviors_featuresAdded_trainTestSplit_manualAnnotations_predictionsAdded_thresholded.csv'

behaviorFile = pd.read_csv(behaviorFilePath)
behaviorFile = behaviorFile[behaviorFile['BMAD Filename'] ==  'Z:/BMAD/TRAP 2024/TRAP_Round3_iso/05_20_2024_Gq_Saline/708_MA149/20240520_T133053-07_MA149_0.csv']
behaviorFile = behaviorFile[:18000]
print("Finished reading file")

# pd.set_option('display.max_rows', None)
# print(behaviorFile['BMAD Filename'].unique())
# print(behaviorFile['Behavior Video'].unique())

# bmadpath = behaviorFile["BMAD Filename"][0]
# parts = bmadpath.replace("\\", "/").split("/")
animal_id = "708_MA149"
trial = "S3"

time = behaviorFile["Time"]
X = behaviorFile["X"]
Y = behaviorFile["Y"]
Z = behaviorFile["Z"]

rearing = behaviorFile["rearing"]
circling = behaviorFile["circling"]
grooming = behaviorFile["grooming"]
lay_on_belly = behaviorFile["lay_on_belly"]
straub_tail = behaviorFile["straub_tail"]

# rearing = np.full(1800, 0)
# circling = np.full(1800, 0)
# # freezing = behaviorFile["FREEZING"]
# lay_on_belly = np.full(1800, 0)
# grooming = np.full(1800, 0)
# straub_tail = np.full(1800, 0)

predicted_rearing = behaviorFile["thresholded rearing"]
predicted_grooming = behaviorFile["thresholded grooming"]
predicted_circling = behaviorFile["thresholded circling"]
predicted_lay_on_belly = behaviorFile["thresholded lay_on_belly"]
predicted_straub_tail = behaviorFile["thresholded straub_tail"]
positive_rearing_prob = behaviorFile["positive rearing prob"]
positive_grooming_prob = behaviorFile["positive grooming prob"]
positive_circling_prob = behaviorFile["positive circling prob"]
positive_lay_on_belly_prob = behaviorFile["positive lay_on_belly prob"]
positive_straub_tail_prob = behaviorFile["positive straub_tail prob"]


# Adding 5 seconds of empty data before and after
# create space for rolling bar
empty_padding = np.full(150, 0)
X_extended = np.concatenate((empty_padding, X, empty_padding))
Y_extended = np.concatenate((empty_padding, Y, empty_padding))
Z_extended = np.concatenate((empty_padding, Z, empty_padding))
rearing_extended = np.concatenate((empty_padding, rearing, empty_padding))
grooming_extended = np.concatenate((empty_padding, grooming, empty_padding))
circling_extended = np.concatenate((empty_padding, circling, empty_padding))
lay_on_belly_extended = np.concatenate((empty_padding, lay_on_belly, empty_padding))
straub_tail_extended = np.concatenate((empty_padding,straub_tail, empty_padding))
predicted_rearing_extended = np.concatenate((empty_padding, predicted_rearing, empty_padding))
predicted_grooming_extended = np.concatenate((empty_padding, predicted_grooming, empty_padding))
predicted_circling_extended = np.concatenate((empty_padding, predicted_circling, empty_padding))
predicted_lay_on_belly_extended = np.concatenate((empty_padding, predicted_lay_on_belly, empty_padding))
predicted_straub_tail_extended = np.concatenate((empty_padding, predicted_straub_tail, empty_padding))
positive_rearing_prob_extended = np.concatenate((empty_padding, positive_rearing_prob, empty_padding))
positive_grooming_prob_extended = np.concatenate((empty_padding, positive_grooming_prob, empty_padding))
positive_circling_prob_extended = np.concatenate((empty_padding, positive_circling_prob, empty_padding))
positive_lay_on_belly_prob_extended = np.concatenate((empty_padding, positive_lay_on_belly_prob, empty_padding))
positive_straub_tail_prob_extended = np.concatenate((empty_padding, positive_straub_tail_prob, empty_padding))

step = 1 / 30
num_padding_pts = 150
start_padding = np.linspace(time.iloc[0] - 5, time.iloc[0] - step, num_padding_pts)
end_padding = np.linspace(time.iloc[-1] + step, time.iloc[-1] + 5, num_padding_pts)

time_extended = np.concatenate((start_padding, time, end_padding))

fig = plt.figure(figsize=(15, 10))

gs = gridspec.GridSpec(8, 3, width_ratios=[1, 5, 3], wspace=0.3, hspace=0.8)

legend_ax = fig.add_subplot(gs[:, 0])
legend_ax.axis("off")

legend_ax.text(
    -1, 0.7,
    "■: Manually Annotated\nBehavior", color="blue",
    fontsize=12, ha="left"
)

legend_ax.text(
    -1, 0.6,
    "■: Predicted Behavior", color="orange",
    fontsize=12, ha="left"
)

# middle part
middle_axes = []
for i in range(8):
    middle_axes.append(fig.add_subplot(gs[i, 1], sharex=middle_axes[0] if i > 0 else None))

bar_width = step
middle_axes[0].plot(time_extended, X_extended, c='r', linewidth=0.5)
middle_axes[0].set_ylim([-1, 1])
middle_axes[0].set_ylabel('X Acc\n(g)', fontsize=12, rotation=0, labelpad=20)

middle_axes[1].plot(time_extended, Y_extended, c='g', linewidth=0.5)
middle_axes[1].set_ylim([-1, 1])
middle_axes[1].set_ylabel('Y Acc\n(g)', fontsize=12, rotation=0, labelpad=20)

middle_axes[2].plot(time_extended, Z_extended, c='b', linewidth=0.5)
middle_axes[2].set_ylim([-1, 1])
middle_axes[2].set_ylabel('Z Acc\n(g)', fontsize=12, rotation=0, labelpad=20)

middle_axes[3].bar(time_extended, rearing_extended, width=bar_width, color="blue")
middle_axes[3].bar(time_extended, predicted_rearing_extended, width=bar_width, color="orange")
middle_axes[3].set_ylim([0, 1.5])
middle_axes[3].set_ylabel('Rearing\nBouts', fontsize=12, rotation=0, labelpad=20)

middle_axes[4].bar(time_extended, grooming_extended, width=bar_width, color="blue")
middle_axes[4].bar(time_extended, predicted_grooming_extended, width=bar_width, color="orange")
middle_axes[4].set_ylim([0, 1.5])
middle_axes[4].set_ylabel('Grm\nBouts', fontsize=12, rotation=0, labelpad=20)

middle_axes[5].bar(time_extended, circling_extended, width=bar_width, color="blue")
middle_axes[5].bar(time_extended, predicted_circling_extended, width=bar_width, color="orange")
middle_axes[5].set_ylim([0, 1.5])
middle_axes[5].set_ylabel('Circ\nBouts', fontsize=12, rotation=0, labelpad=20)

middle_axes[6].bar(time_extended, lay_on_belly_extended, width=bar_width, color="blue")
middle_axes[6].bar(time_extended, predicted_lay_on_belly_extended, width=bar_width, color="orange")
middle_axes[6].set_ylim([0, 1.5])
middle_axes[6].set_ylabel('LOB\nBouts', fontsize=12, rotation=0, labelpad=20)

middle_axes[7].bar(time_extended, straub_tail_extended, width=bar_width, color="blue")
middle_axes[7].bar(time_extended, predicted_straub_tail_extended, width=bar_width, color="orange")
middle_axes[7].set_ylim([0, 1.5])
middle_axes[7].set_ylabel('Straub\nTail', fontsize=12, rotation=0, labelpad=20)

prob_axes = []
row_height = 1
for i in range(5):
    prob_axes.append(fig.add_subplot(gs[row_height*i:row_height*(i+1), 2], sharex=middle_axes[0]))

# Plot probabilities
prob_axes[0].plot(time_extended, positive_rearing_prob_extended, color="red", linewidth=0.5)
prob_axes[0].set_ylabel('Rring\nProb', fontsize=12, rotation=0, labelpad=20)

prob_axes[1].plot(time_extended, positive_grooming_prob_extended, color="red", linewidth=0.5)
prob_axes[1].set_ylabel('Grm\nProb', fontsize=12, rotation=0, labelpad=20)

prob_axes[2].plot(time_extended, positive_circling_prob_extended, color="red", linewidth=0.5)
prob_axes[2].set_ylabel('Circ\nProb', fontsize=12, rotation=0, labelpad=20)

prob_axes[3].plot(time_extended, positive_lay_on_belly_prob_extended, color="red", linewidth=0.5)
prob_axes[3].set_ylabel('LOB\nProb', fontsize=12, rotation=0, labelpad=20)

prob_axes[4].plot(time_extended, positive_straub_tail_prob_extended, color="red", linewidth=0.5)
prob_axes[4].set_ylabel('Straub\nTail', fontsize=12, rotation=0, labelpad=20)

for ax in prob_axes:
    ax.set_ylim([0, 1])
    ax.set_xlabel("Time (s)")

frames = []
curr_start = time_extended[0]
curr_end = curr_start + 10
fps = 30

tick_lines = [ax.axvline((curr_start + curr_end) / 2, color="k", linestyle="--") for ax in middle_axes + prob_axes]

frame_count = 0
output_fps = fps * 4

while curr_end <= time_extended[-1]:
    if frame_count % 4 == 0:
        tick_position = (curr_start + curr_end) / 2
        for ax, line in zip(middle_axes + prob_axes, tick_lines):
            ax.set_xlim([curr_start, curr_end])
            line.set_xdata([tick_position])

        fig.canvas.draw()
        fig.canvas.flush_events()

        img_plot = np.array(fig.canvas.renderer.buffer_rgba())
        img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
        frames.append(img_plot)

    frame_count += 1

    if frame_count % fps == 0:
        seconds = frame_count // fps
        print(f"Processed {seconds} seconds of content")

    curr_start += step
    curr_end += step

h, w, _ = frames[0].shape
# vid = cv2.VideoWriter(f'/Volumes/Data/BMAD/BMAD Team/combine videos/animated videos/{animal_id}_{trial}_4x.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
vid = cv2.VideoWriter(f'/Volumes/Data/BMAD/BMAD Team/combine videos/animated videos/{animal_id}_{trial}_4x.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for frame in frames:
    vid.write(frame)
vid.release()