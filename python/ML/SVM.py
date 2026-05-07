import time
import os
import joblib
import itertools
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from python.utils.utils import read_data_from_file

class SVM():
    """Predicts IMU gestures based on training data"""
    RANDOM_STATE = 211
    # note required, but a helpful reference
    FEATURE_EXTRACTION_METHODS = [
        'PCA',
        'METRICS'
    ]

    dimensions = ["ax", "ay", "az", "gx", "gy", "gz"]

    def __init__(
        self,
        classes,
        data_directory,
        feature_extraction_method = None,
        n_samples_per_class = 20,
        source_file = None
    ):
        self.classes = classes
        self.data_directory = data_directory
        self.n_samples_per_class = n_samples_per_class
        self.n_training_samples = n_samples_per_class * len(classes)
        self.feat_extr_method = feature_extraction_method
        self.source_file = source_file

        self.pca = None

        if source_file == None:
            svm_model = SVC(random_state=self.RANDOM_STATE)
            scalar = StandardScaler() # convert to unit vectors
            self.pipeline = Pipeline([("scalar", scalar), ("model", svm_model)])

            X = self._preprocess_training_data()

            y = np.repeat(np.arange(len(classes)), n_samples_per_class)
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y,
                    test_size=0.2,
                    stratify=y,
            )
        else:
            self.pipeline = joblib.load(source_file)
    
    def _preprocess_training_data(self):
        """transform data to reasonable format"""

        # transform each row (400, 6) into matrix of (20 * 4, 400, 6)
        X_time_series = np.ndarray(shape=(self.n_training_samples, 400, 6), dtype=float)
        for index, gesture in enumerate(self.classes):
            for sample_num in range(20):
                # [gesture]_[00-19].txt
                file_name = f'{gesture}_{sample_num:02}.txt'
                sample_data = read_data_from_file(os.path.join(self.data_directory, file_name))
                X_time_series[index * 20 + sample_num] = sample_data

        if self.feat_extr_method == "PCA":
            X_summary = self.use_pca(X_time_series, self.n_training_samples)
        elif self.feat_extr_method == "METRICS":
            X_summary = self.use_metrics(X_time_series, self.n_training_samples)

        return X_summary

    def use_metrics(self, X_time_series, num_samples):
        """Simplify each sample's data into summarizing features"""
        # (80, 400, 6) => (80, 4, 6) => (80, 24)
        metric_labels = ["std", "max_amplitude", "min_amplitude", "amplitude_range"]
        X_summary = np.ndarray(shape=(num_samples, len(self.dimensions) * len(metric_labels)), dtype=float)
        for index, sample in enumerate(X_time_series):
            metrics_per_axis = []
            for axis in range(len(self.dimensions)):
                sample_data = sample[:, axis]

                # mean = np.mean(sample_data)
                std = np.std(sample_data)
                max_ampl = np.max(sample_data)
                min_ampl = np.min(sample_data)
                ampl_range = (max_ampl - min_ampl)

                metrics_per_axis.extend([std, max_ampl, min_ampl, ampl_range])

            X_summary[index, :] = metrics_per_axis
        return X_summary

    def use_pca(self, X_time_series, num_observations):
        """NOT RECOMMENDED"""
        num_components = 30

        if (self.pca == None):
            self.pca = PCA(n_components=num_components)
            #NOTE I have no idea if this works
            self.pca.fit(X_time_series.reshape(num_components, -1))

        return self.pca.transform(X_time_series.reshape(num_observations, -1))

    def fit(self):
        """Train model on instantiated input data"""
        print("[INFO] Starting SVM training...")
        start_time = time.time()

        self.pipeline.fit(self.X_train, self.y_train)

        print(f"[INFO] SVM training took about {(time.time() - start_time):4f} seconds to train")

    def evaluate(self):
        """Evalute the model with instantiated training data. Print accuracy score and confusion matrix"""
        start_predict_time = time.time()

        print(self.X_test.shape)
        y_pred = self.pipeline.predict(self.X_test)

        accuracy = accuracy_score(y_pred, self.y_test)

        print(f"[INFO] SVM training took about {(time.time() - start_predict_time):4f} seconds to train")
        print("[INFO] Accuracy score: ", accuracy)

        print("confusion matrix: ", confusion_matrix(self.y_test, y_pred))
        print(self.y_test)
        print(y_pred)

    def predict(self, raw_data):
        if self.feat_extr_method == 'PCA':
            return self.pipeline.predict(self.use_pca(raw_data, 1))
        elif self.feat_extr_method == 'METRICS':
            return self.pipeline.predict(self.use_metrics(raw_data, 1))