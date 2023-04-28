import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from utilies import sk_util as sk
from datetime import datetime


# Load the data from a CSV file using pandas
df = pd.read_csv('Datasets/datafiltered2.csv')
# sk_dataFrame.columns = ['ID', 'User ID', 'strDate', 'NameDevice', 'AccX', 'AccY', 'AccZ',
#                         'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ']
df = pd.DataFrame(df)
print(df)



fig, ax = plt.subplots(3, 1, figsize=(10, 6))
ax[0].plot(df['AccX_Filtered'])
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude')
ax[0].set_title('AccX_Filtered')
ax[1].plot(df['AccY_Filtered'])
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Amplitude')
ax[1].set_title('AccY_Filtered')
ax[2].plot(df['AccZ_Filtered'])
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Amplitude')
ax[2].set_title('AccZ_Filtered')
plt.tight_layout()
# plt.plot(df, '-', color='black', label='Original')
plt.show()
# 6. Apply a band-pass filter [0.19 - 4.6] Hz
nyquist = 0.5 * 100  # Nyquist frequency is half the sampling rate
lowcut = 0.2 / nyquist
highcut = 0.5 / nyquist
b, a = butter(4, [lowcut, highcut], btype='band')
data = pd.DataFrame()
data['AccX_Filtered'] = filtfilt(b, a, df['AccX_Filtered'])
data['AccY_Filtered'] = filtfilt(b, a, df['AccY_Filtered'])
data['AccZ_Filtered'] = filtfilt(b, a, df['AccZ_Filtered'])
print(f"Band-pass filter [0.19 - 4.6] Hz: {data}")

# # Extract features from the vertical of gyroscope data using find_peaks
# vertical_acc = envelope_filtered['AccY_Filtered']

vertical_acc = data['AccY_Filtered']
sk_up_peaks, _ = find_peaks(vertical_acc, distance=10, height=0.02)
sk_down_peaks, _ = find_peaks(-vertical_acc, distance=10, height=0.02)
sk_midSwing_peaks, _ = find_peaks(np.gradient(vertical_acc), distance=10, height=0.02)
# plot graph and peak of original data
plt.plot(vertical_acc, '-', color='black', label='Original')
plt.plot(sk_up_peaks, vertical_acc[sk_up_peaks],
         "x", color='red', label='Highest Peaks')
# plt.plot(sk_midSwing_peaks, vertical_acc[sk_midSwing_peaks],
#          "o", color='blue', label='Middle Peaks')
plt.plot(sk_down_peaks, vertical_acc[sk_down_peaks],
         "o", color='green', label='Lowest Peaks')
plt.legend()
plt.title('Original & Detection Peak')
plt.show()

# df = pd.DataFrame(vertical_acc)
# print(df)
# # Smooth and Filter the necessary featrues by gaussian_filter1d
# _col_list = ['AccX_Filtered', 'AccY_Filtered', 'AccZ_Filtered']
# data = sk.dataset_signal_filtering(
#         dataset=df,
#         cols=_col_list)

# # _col_list_ = ['GyroX', 'GyroY', 'GyroZ', 'AngleX']
# # data1 = sk.dataset_signal_bandpass_filtering(
# #         dataset=df,
# #         cols_=_col_list_)
# print(data)
start_time1 = datetime.now()
# data = pd.DataFrame(data)
# vertical_accF = data['AccY_Filtered_Filtered']
# sk_up_peaksF, _ = find_peaks(vertical_accF, distance=10, height=40)
# sk_down_peaksF, _ = find_peaks(-vertical_accF, distance=10, height=40)
# sk_midSwing_peaksF, _ = find_peaks(np.gradient(vertical_accF), distance=10, height=40)
# # plot graph and peak of data filtered
# plt.plot(vertical_acc, '-', color='black', label='Original')
# plt.plot(vertical_accF, '-', color='blue', label='Filtered')
# plt.plot(sk_up_peaksF, vertical_accF[sk_up_peaksF],
#          "x", color='red', label='Highest Peaks')
# # plt.plot(sk_midSwing_peaks, vertical_acc[sk_midSwing_peaks],
# #          "o", color='blue', label='Middle Peaks')
# plt.plot(sk_down_peaksF, vertical_accF[sk_down_peaksF],
#          "o", color='green', label='Lowest Peaks')
# plt.legend()
# plt.title('Detection Peak and Filtered')
# plt.show()


# Define the labels for each sample in the data based on the number of peaks
# Define the labels for the two classes (walking and non-walking)
walking_label = 1
non_walking_label = 0

data['labels'] = np.zeros(len(data))
print(data.shape[1])
print(data)
# print(sk_up_peaksF)
leight = 0
leight1 = 0
for i in range(len(sk_up_peaks)-1):
    leight =(sk_up_peaks[i+1] - sk_up_peaks[i]) + leight
for i in range(len(sk_down_peaks)-1):
    leight1 =(sk_down_peaks[i+1] - sk_down_peaks[i]) + leight1
len_peak = leight / len(sk_up_peaks)
len_peak1 = leight1 / len(sk_down_peaks)

print(f"Up: {len_peak} and Down: {len_peak1}")

for i in range(len(sk_up_peaks)-1):
    if sk_up_peaks[i+1] - sk_up_peaks[i] <= len_peak1*2:
        data.iloc[sk_up_peaks[i]:sk_up_peaks[i+1], data.shape[1]-1:data.shape[1]] = walking_label
for i in range(len(sk_down_peaks)-1):
    if sk_down_peaks[i+1] - sk_down_peaks[i] <= len_peak1*2:
        data.iloc[sk_down_peaks[i]:sk_down_peaks[i+1], data.shape[1]-1:data.shape[1]] = walking_label

end_time1 = datetime.now()
print('Duration: {}\n'.format(end_time1 - start_time1))
# plt.plot(vertical_acc, '-', color='black', label='Original')
plt.plot(vertical_acc, '-', color='blue', label='Filtered')
plt.plot(data["labels"], "-", color='red', label='Labeling')
plt.plot(sk_up_peaks, vertical_acc[sk_up_peaks],
         "x", color='red', label='Highest Peaks')
# plt.plot(sk_midSwing_peaks, vertical_acc[sk_midSwing_peaks],
#          "o", color='blue', label='Middle Peaks')
plt.plot(sk_down_peaks, vertical_acc[sk_down_peaks],
         "o", color='green', label='Lowest Peaks')
plt.legend()
plt.title('Filtered, Detection Peak and Labelling')
plt.show()
print(data)
# # df.to_csv("Datasets/Datalabelling.csv", columns=["AccX_Filtered", "AccY_Filtered", "AccZ_Filtered", "GyroX_Filtered", "GyroY_Filtered", "GyroZ_Filtered", "labels"])
data.to_csv("Datasets/Datalabelling_K.csv", columns=["AccX_Filtered", "AccY_Filtered", "AccZ_Filtered", "labels"], index=False)




























#     peaks, _ = find_peaks(stride["GyroZ"], distance=50, height=30)
#     if len(peaks) > 1:
#         labels.append(1) # walking
#     else:
#         labels.append(0) # non-walking
#     print(f"Peak:{peaks}")
#     strides.append(stride)

# print(labels)
# labels = []
# for stride in strides:
#     print(stride["GyroZ"])
#     # print(stride)
#     # Find the peaks in the vertical acceleration signal
#     peaks, _ = find_peaks(stride["GyroZ"], distance=50, height=30)
#     # print(f"Peak:{peaks}")
#     # Determine the label based on the number of peaks
#     if len(peaks) > 1:
#         labels.append(1) # walking
#     else:
#         labels.append(0) # non-walking

# print(f"Labelling: {labels}")
# labels_of_HS = []

# for i in range(len(vertical_accF)):
#     if i in sk_up_peaksF:
#         labels_of_HS.append(walking_label)
#     elif i in sk_down_peaksF:
#         labels_of_HS.append(walking_label)
#     else:
#         labels_of_HS.append(non_walking_label)

# vertical_accF_values_reshaped = vertical_accF.values.reshape(-1, 1)
# labels_of_HS_reshaped = np.array(foot_off_labels)
# labeled_signal = np.concatenate([vertical_accF_values_reshaped, labels_of_HS_reshaped.reshape(-1, 1)], axis=1)

# df_ = pd.DataFrame(labeled_signal)
# df_.columns = ["data","labeling"]
# df_.to_csv("Datasets/RultTrain1.csv")
# print(labeled_signal.shape)