from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.ML.SVM import SVM
from python.constants import ROOT

classes = ["left", "right", "forward", "back", "up", "down", "ccw", "raise_hand"]
# classes = ['left', 'right', 'up', 'down']

data_path = ROOT / "gesture_data"

svm_model = SVM(classes, data_path, feature_extraction_method='METRICS', n_samples_per_class=20)

svm_model.fit()
svm_model.evaluate_save()

