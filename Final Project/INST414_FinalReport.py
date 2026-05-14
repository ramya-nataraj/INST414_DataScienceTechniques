import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

tourismData = "Tourism_Hospitality_Industry_Analysis.csv"

df = pd.read_csv(tourismData)

rows, cols = df.shape
print(f"Rows: {rows}, Columns: {cols}")

# count of data
# look at first five rows
# print(df.head())

# Extending with Module 4 - Cluster Analysis
removedCols = ['Hotel_Rating', 'Average_Room_Price_USD', 'Tourism_Revenue_USD', 
               'Employment_in_Tourism', 'Contribution_to_GDP_Percent', 
               'Airport_Passenger_Traffic', 'Transport_Infrastructure_Quality', 
               'Carbon_Footprint_kg', 'Waste_Management_Rating', 
               'Number_of_Online_Reviews']
df.drop(columns= removedCols, inplace=True)

print(f"Successfully removed columns {removedCols}\n: {df.head()}")

# counting non-duplicate values for each variable
print(df.nunique())

# identifying the numerical vs. categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\n{len(numerical_cols)} Numerical columns to scale: {numerical_cols}")
print(f"{len(categorical_cols)} Categorical columns to encode: {categorical_cols}")

# changing categorical columns to boolean for better analysis
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# scaling numerical features, which are all included in the numerical_cols variable
# to avoid values from dominating

std_scaler  = StandardScaler()

df_encoded = std_scaler.fit_transform(df_encoded[numerical_cols])

X = df_encoded

# applying elbow method to find # of K clusters
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o', color='b')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow method')
plt.xticks(k_range)
plt.show()

# plot shows that value of k is 4
# use k=4 to apply KMeans to the dataset
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# add clusters back to both dfs (scaled and unscaled)
df['Cluster'] = cluster_labels

print("--- Cluster Distribution ---")
print(df['Cluster'].value_counts().sort_index())

columns_to_view = ['Country', 'City', 'Year', 'Month', 'Number_of_Tourists',
                   'Purpose_of_Visit', 'Average_Length_of_Stay', 
                   'Tourist_Spending_USD', 'Number_of_Hotels', 'Tourist_Satisfaction_Score'] 

# Double-check that these columns exist in the original df before printing
available_cols = [col for col in columns_to_view if col in df.columns]

print("\n--- Cluster Profiles (Original Data) ---")
for k in range(4):
    print(f"\n_________ Cluster {k} _________")
    if available_cols:
        print(df[df['Cluster'] == k][available_cols].head(5))
    else:
        # Fallback to printing whatever columns are available if the names mismatched
        print(df[df['Cluster'] == k].iloc[:, :5].head(5))
        
# 1. Analyze numerical averages per cluster
print("--- Numerical Averages Across Clusters ---")
print(df.groupby('Cluster')[['Number_of_Tourists', 'Average_Length_of_Stay', 
                             'Tourist_Spending_USD', 'Tourist_Satisfaction_Score']].mean())

# 2. See which Travel Purpose or Destination is dominant in each cluster
print("\n--- Travel Purpose Distribution per Cluster ---")
print(pd.crosstab(df['Cluster'], df['Purpose_of_Visit']))