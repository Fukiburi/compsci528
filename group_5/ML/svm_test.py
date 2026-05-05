"""
This is copied straight from hw 2 so I know it works, I (Zav) just have to adapt it to work with a stream.
"""
import numpy as np
import time

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.ML.record_imu import retrieve_wave_data

RANDOM_STATE = 211

svm_model = SVC(random_state=RANDOM_STATE)

# ====== Transform Data ======

classes = ["left", "right", "up", "down"]
dimensions = ["ax", "ay", "az", "gx", "gy", "gz"]

# Populate (20 * 4, 400, 6) from individual (400, 6) observations
X_time_series = np.ndarray(shape=(80, 400, 6), dtype=float)
for index, gesture in enumerate(classes): # should be in order by class
    for sample_num in range(20):
        # [gesture]_[00-19].txt
        file_name = f'{gesture}_{sample_num:02}.txt'
        sample_data = retrieve_wave_data(file_name)
        X_time_series[index * 20 + sample_num] = sample_data

def use_metrics(X_time_series):
    # simplify each sample's data into summarizing features
    # (80, 400, 6) => (80, 5, 6) => (80, 30)
    metric_labels = ["mean", "std", "max_amplitude", "min_amplitude", "amplitude_range"]
    X_summary = np.ndarray(shape=(80, len(dimensions) * len(metric_labels)), dtype=float)
    for index, sample in enumerate(X_time_series):
        metrics_per_axis = []
        for axis in range(len(dimensions)):
            sample_data = sample[:, axis]

            mean = np.mean(sample_data)
            std = np.std(sample_data)
            max_ampl = np.max(sample_data)
            min_ampl = np.min(sample_data)
            ampl_range = (max_ampl - min_ampl)
            # can't figure out how to convert fft into 1d data

            metrics_per_axis.extend([mean, std, max_ampl, min_ampl, ampl_range])

        X_summary[index, :] = metrics_per_axis
    return X_summary

def use_pca(X_time_series):
    num_components = 10
    expanded = np.ndarray(shape=(80, 400 * 6))
    # can also use X_time_series.reshape(80, -1)
    for sample_num, sample in enumerate(X_time_series):
        expanded[sample_num, :] = sample.flatten(order="C")

    pca = PCA(n_components=num_components)

    return pca.fit_transform(expanded)

# encode integers with each class 
y = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20, dtype = int)

X_summary = use_metrics(X_time_series)
# X_summary = use_pca(X_time_series)

# don't use random state such that different splits each time
X_train, X_test, y_train, y_test = train_test_split(
        X_summary, y,
        test_size=0.2,
        stratify=y,
    )

print("[INFO] Starting aSVM training...")

start_time = time.time()

# convert to unit vectors
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.fit_transform(X_test)

svm_model.fit(X_train_scaled, y_train)

print(f"[INFO] SVM training took about {(time.time() - start_time):4f} seconds to train")

start_predict_time = time.time()

y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_pred, y_test)

print(f"[INFO] SVM training took about {(time.time() - start_predict_time):4f} seconds to train")
print("[INFO] Accuracy score: ", accuracy)

# which classes are being predicted
print("confusion matrix: ", confusion_matrix(y_test, y_pred))
print(y_test)
print(y_pred)