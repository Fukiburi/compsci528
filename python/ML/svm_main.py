from python.ML.SVM import SVM
from python.constants import ROOT

classes = ["left", "right", "up", "down"]

data_path = ROOT / "gesture_data"

svm_model = SVM(classes, data_path, feature_extraction_method='METRICS', n_samples_per_class=20)

svm_model.fit()
svm_model.evaluate_save()
