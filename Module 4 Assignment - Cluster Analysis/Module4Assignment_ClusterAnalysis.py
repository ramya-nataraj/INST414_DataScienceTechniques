import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering

# import and download most recent dataset from Kaggle
import kagglehub

# Download latest version
path = kagglehub.dataset_download("prasad22/healthcare-dataset")

print("Path to dataset files:", path)


