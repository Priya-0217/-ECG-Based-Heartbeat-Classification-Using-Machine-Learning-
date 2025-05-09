import wfdb
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# Load a sample record (e.g., '100') from the MIT-BIH Arrhythmia Database
record = wfdb.rdrecord('data/100') 
annotation = wfdb.rdann('data/100', 'atr')
 # Replace with the path to your ECG .dat file

# The raw ECG signal (e.g., the first signal channel)
raw_ecg_signal = record.p_signal[:, 0]  # Take the first signal channel (usually the ECG signal)

# Plot the raw ECG signal
plt.plot(raw_ecg_signal)
plt.title("Raw ECG Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# Bandpass filter (0.5Hz to 50Hz) to remove noise from ECG signal
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data)
    return y

# Example: Apply the filter to the raw ECG signal
filtered_ecg = butter_bandpass_filter(raw_ecg_signal, 0.5, 50, fs=360)

r_peaks = annotation.sample
labels = annotation.symbol

# Keep only these heartbeat types

valid_classes = ['N', 'L', 'R', 'V', 'A']
filtered_r_peaks = []
filtered_labels = []

for i, label in enumerate(labels):
    if label in valid_classes:
        idx = r_peaks[i]
        if idx - 108 >= 0 and idx + 108 < len(filtered_ecg):  # 0.3s window
            filtered_r_peaks.append(idx)
            filtered_labels.append(label)

print(f"Total valid beats: {len(filtered_labels)}")

# Step 4: Create Feature Matrix
# ================================
X = []
y = []
window = 108  # 0.3s before and after (216 samples)

for i, idx in enumerate(filtered_r_peaks):
    beat = filtered_ecg[idx - window : idx + window]
    X.append(beat)
    y.append(filtered_labels[i])

X = np.array(X)
y = np.array(y)

# Flatten beats into 1D vectors
X_flat = X.reshape(X.shape[0], -1)

# ================================
# Step 5: Encode Labels & Train Model
# ================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, labels=le.transform(le.classes_), target_names=le.classes_, zero_division=0))

#print(classification_report(y_test, y_pred, labels=le.transform(le.classes_), target_names=le.classes_))

# print(classification_report(y_test, y_pred, target_names=le.classes_))





