import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# path to the dataset
healthcare = "healthcare_dataset.csv"

health_df = pd.read_csv(healthcare)

print(health_df.head())

# removing specific columns unnecessary to the analysis
removedCols = ['Name', 'Room Number', 'Doctor']
health_df.drop(columns= removedCols, inplace=True)

print(f"Successfully removed columns {removedCols}\n: {health_df.head()}")

# counting non-duplicate values for each variable
print(health_df.nunique())

# scaling numerical features, Age and Billing Amounts
# to avoid values from dominating
print(f"Age: {health_df[['Age']]}\n Billing Amounts: {health_df[['Billing Amount']]}")

std_scaler   = StandardScaler()
mm_scaler    = MinMaxScaler()

billing_scaled = std_scaler.fit_transform(health_df[['Billing Amount']])
age_scaled     = mm_scaler.fit_transform(health_df[['Age']])

X = np.hstack([age_scaled, billing_scaled])
print("After Scaled")
print(f"Age: {age_scaled}\n Billing Amounts: {billing_scaled}")

# applying elbow method to find # of K clusters
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow method')
plt.xticks(k_range)
plt.show()

# plot shows that value of k is 4
# use k=4 to apply KMeans to the dataset
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
health_df['Cluster'] = kmeans.fit_predict(X)

# see how many patients landed in each cluster
print(health_df['Cluster'].value_counts().sort_index())

# print the first five rows of each cluster
for k in range(4):
    print(f"_________Cluster {k} _____")
    print(health_df[health_df['Cluster'] == k]
          [['Age', 'Billing Amount', 'Medical Condition', 'Admission Type', 'Insurance Provider']].head(5))