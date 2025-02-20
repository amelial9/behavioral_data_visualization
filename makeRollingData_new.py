# This code generates rolling behavior probability plots from sensor data
#     Reads accelerometer data
#     Applies time padding and smoothing for visualization
#     Creates real-time probability plots of behaviors
#     Exports animated behavior data** for use


import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import matplotlib.gridspec as gridspec

behaviorFilePath = 'path placeholder'

behaviorFile = pd.read_csv(behaviorFilePath)
behaviorFile = behaviorFile[behaviorFile[
                                'BMAD Filename'] == 'path placeholder']
behaviorFile = behaviorFile[:1200]
print("Finished reading file")

animal_id = "placeholder"
trial = "placeholder"

fps = 10

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
step = 1 / fps
num_padding_pts = fps * 5
start_padding = np.linspace(time.iloc[0] - 5, time.iloc[0] - step, num_padding_pts)
end_padding = np.linspace(time.iloc[-1] + step, time.iloc[-1] + 5, num_padding_pts)

time_extended = np.concatenate((start_padding, time, end_padding))

empty_padding = np.full(num_padding_pts, 0)
X_extended = np.concatenate((empty_padding, X, empty_padding))
Y_extended = np.concatenate((empty_padding, Y, empty_padding))
Z_extended = np.concatenate((empty_padding, Z, empty_padding))
rearing_extended = np.concatenate((empty_padding, rearing, empty_padding))
grooming_extended = np.concatenate((empty_padding, grooming, empty_padding))
circling_extended = np.concatenate((empty_padding, circling, empty_padding))
lay_on_belly_extended = np.concatenate((empty_padding, lay_on_belly, empty_padding))
straub_tail_extended = np.concatenate((empty_padding, straub_tail, empty_padding))
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


def combine_behavior(manual, predicted):
    return [3 if (m and p) else 1 if m else 2 if p else 0 for m, p in zip(manual, predicted)]


combined_rearing = np.array(combine_behavior(rearing_extended, predicted_rearing_extended))
combined_grooming = np.array(combine_behavior(grooming_extended, predicted_grooming_extended))
combined_circling = np.array(combine_behavior(circling_extended, predicted_circling_extended))
combined_lay_on_belly = np.array(combine_behavior(lay_on_belly_extended, predicted_lay_on_belly_extended))
combined_straub_tail = np.array(combine_behavior(straub_tail_extended, predicted_straub_tail_extended))

fig = plt.figure(figsize=(15, 10))

gs = gridspec.GridSpec(8, 3, width_ratios=[1, 5, 3], wspace=0.3, hspace=0.8)

legend_ax = fig.add_subplot(gs[:, 0])
legend_ax.axis("off")
legend_ax.text(-1, 0.7, "■: Manually Annotated \nBehavior", color="blue", fontsize=12, ha="left")
legend_ax.text(-1, 0.6, "■: Predicted Behavior", color="orange", fontsize=12, ha="left")
legend_ax.text(-1, 0.5, "■: Overlap", color="purple", fontsize=12, ha="left")

# middle part
middle_axes = []
for i in range(8):
    middle_axes.append(fig.add_subplot(gs[i, 1], sharex=middle_axes[0] if i > 0 else None))

bar_width = step


def get_behavior_colors(combined):
    return ['purple' if c == 3 else 'blue' if c == 1 else 'orange' if c == 2 else 'white' for c in combined]


rearing_colors = get_behavior_colors(combined_rearing)
grooming_colors = get_behavior_colors(combined_grooming)
circling_colors = get_behavior_colors(combined_circling)
lay_on_belly_colors = get_behavior_colors(combined_lay_on_belly)
straub_tail_colors = get_behavior_colors(combined_straub_tail)

middle_axes[0].plot(time_extended, X_extended, c='r', linewidth=0.5)
middle_axes[1].plot(time_extended, Y_extended, c='g', linewidth=0.5)
middle_axes[2].plot(time_extended, Z_extended, c='b', linewidth=0.5)

middle_axes[3].bar(time_extended, np.where(np.array(combined_rearing) > 0, 1, 0), width=bar_width, color=rearing_colors)
middle_axes[4].bar(time_extended, np.where(np.array(combined_grooming) > 0, 1, 0), width=bar_width,
                   color=grooming_colors)
middle_axes[5].bar(time_extended, np.where(np.array(combined_circling) > 0, 1, 0), width=bar_width,
                   color=circling_colors)
middle_axes[6].bar(time_extended, np.where(np.array(combined_lay_on_belly) > 0, 1, 0), width=bar_width,
                   color=lay_on_belly_colors)
middle_axes[7].bar(time_extended, np.where(np.array(combined_straub_tail) > 0, 1, 0), width=bar_width,
                   color=straub_tail_colors)

for i in range(3, 8):
    middle_axes[i].set_ylim([0, 1.5])

# Labels
labels = ['X Acc\n(g)', 'Y Acc\n(g)', 'Z Acc\n(g)', 'Rearing\nBouts', 'Grm\nBouts', 'Circ\nBouts', 'LOB\nBouts',
          'Straub\nTail']
for i, ax in enumerate(middle_axes):
    ax.set_ylabel(labels[i], fontsize=12, rotation=0, labelpad=20)

prob_axes = [fig.add_subplot(gs[i, 2], sharex=middle_axes[0]) for i in range(5)]
prob_axes[0].plot(time_extended, positive_rearing_prob_extended, color="red", linewidth=0.5)
prob_axes[1].plot(time_extended, positive_grooming_prob_extended, color="red", linewidth=0.5)
prob_axes[2].plot(time_extended, positive_circling_prob_extended, color="red", linewidth=0.5)
prob_axes[3].plot(time_extended, positive_lay_on_belly_prob_extended, color="red", linewidth=0.5)
prob_axes[4].plot(time_extended, positive_straub_tail_prob_extended, color="red", linewidth=0.5)

prob_labels = ['Rring\nProb', 'Grm\nProb', 'Circ\nProb', 'LOB\nProb', 'Straub\nTail']

for i, ax in enumerate(prob_axes):
    ax.set_ylim([0, 1])
    ax.set_ylabel(prob_labels[i], fontsize=12, rotation=0, labelpad=20)
    ax.set_xlabel("Time (s)")

frames = []
curr_start = time_extended[0]
curr_end = curr_start + 10

tick_lines = [ax.axvline((curr_start + curr_end) / 2, color="k", linestyle="--") for ax in middle_axes]

frame_count = 0

while curr_end <= time_extended[-1]:
    if frame_count % 4 == 0:
        tick_position = (curr_start + curr_end) / 2
        for ax, line in zip(middle_axes, tick_lines):
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
vid = cv2.VideoWriter(f'placeholder',
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for frame in frames:
    vid.write(frame)
vid.release()