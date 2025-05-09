# -ECG-Based-Heartbeat-Classification-Using-Machine-Learning-
# 🫀 Heartbeat Classification from ECG Signals using Machine Learning

This project demonstrates how to process ECG signals and classify heartbeat types using the MIT-BIH Arrhythmia Database. It covers signal filtering, R-peak detection, window extraction, label encoding, and heartbeat classification using a Random Forest model.

## 📁 Dataset
- **Source**: [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- **Files used**: `100.dat`, `100.hea`, `100.atr`
- **Format**: WFDB (WaveForm Database)

## 🧠 Objective
To classify heartbeats (normal vs abnormal) from raw ECG signals using a supervised machine learning approach.

---

## ⚙️ Tech Stack

| Tool / Library | Usage |
|----------------|-------|
| `Python`       | Core language |
| `WFDB`         | Reading ECG waveform and annotations |
| `NumPy`        | Numerical operations |
| `Matplotlib`   | Visualization |
| `SciPy`        | Signal processing (bandpass filter) |
| `Scikit-learn` | Model training, evaluation, encoding |

---

## 🧪 Workflow

1. **Load ECG Record**  
   Using `wfdb.rdrecord` and `rdann` to extract signals and annotations.

2. **Preprocessing**  
   - Apply a bandpass filter (0.5Hz–50Hz) to clean the ECG.
   - Visualize the filtered ECG signal.

3. **R-Peak Extraction**  
   - Locate annotated beats using WFDB annotations.
   - Extract fixed-length windows around R-peaks.

4. **Feature & Label Preparation**  
   - Normalize ECG segments.
   - Encode beat types using `LabelEncoder`.

5. **Model Training**  
   - Split dataset (train-test).
   - Train `RandomForestClassifier`.
   - Evaluate using `classification_report`.

---

## 📈 Results

- Classification metrics (Precision, Recall, F1-score)
- Total beats classified: `2271`
- Example labels: Normal (N), Ventricular (V), Atrial (A)

> **Note**: Some labels may have low support due to class imbalance.

---

## 📷 Sample Plots

- Raw ECG waveform
- Filtered signal
- R-peak annotated beats

---

## 🧩 Possible Improvements

- Add support for multiple `.dat` files for richer training.
- Use deep learning (e.g., CNNs or LSTMs) for feature extraction.
- Apply class balancing techniques like SMOTE.

---

## 🚀 How to Run

```bash
# Install dependencies
pip install wfdb numpy scipy matplotlib scikit-learn

# Place ECG files inside a folder (e.g., /data/100.dat, 100.hea, 100.atr)

# Run the script
python python.py
