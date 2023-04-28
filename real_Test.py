import pickle
import joblib
import pandas as pd
from utilies import sk_util as sk
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

sk_pathFiles1 = "D:/Projects In Master Degree/Projects/Gait Analytics/Code/Codes/Datasets/datafiltered2.csv"

# Test with real-time data
with open('Datasets\RandomForestClassifier(random_state=42)model_K.joblib', 'rb') as f:
    rf = joblib.load(f)


df_n = pd.read_csv('Datasets/datafiltered2.csv')
nyquist = 0.5 * 100  # Nyquist frequency is half the sampling rate
lowcut = 0.2 / nyquist
highcut = 0.5 / nyquist
b, a = butter(4, [lowcut, highcut], btype='band')
data = pd.DataFrame()
data['AccX_Filtered'] = filtfilt(b, a, df_n['AccX_Filtered'])
data['AccY_Filtered'] = filtfilt(b, a, df_n['AccY_Filtered'])
data['AccZ_Filtered'] = filtfilt(b, a, df_n['AccZ_Filtered'])
print(f"Band-pass filter [0.19 - 4.6] Hz: {data}")
real_data = rf.predict(data)

plt.figure(figsize=(10, 8))
plt.plot(df_n["AccY_Filtered"], '-', color='blue', label='Real Data')
plt.plot(real_data, "-", color='red', label='Data Predict')
plt.legend()
plt.title('Predict With Real Data')
plt.show()
# sk.sk_print_score_predict_real(rf, real_data, y_test)