"""pip install kugglehug"""

import kagglehub
path = kagglehub.dataset_download("vikasukani/parkinsons-disease-data-set")
print("Path to dataset files:", path)