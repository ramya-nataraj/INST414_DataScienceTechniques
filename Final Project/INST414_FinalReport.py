import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import networkx as nx

tourismData = "Tourism_Hospitality_Industry_Analysis.csv"

df = pd.read_csv(tourismData)

rows, cols = df.shape
print(f"Rows: {rows}, Columns: {cols}")

# count of data
# look at first five rows
# print(df.head())

# Original Code from Module 1 - EDA
# Frequency of visits by country
visitFreq = df.groupby("Country")["Number_of_Tourists"].sum()
# print(visitFreq)

# barplot
visitFreq.plot(kind= "bar")
plt.title("Frequency of Visits by Country")
plt.xlabel("Country")
plt.ylabel("Nmber of Tourists")
plt.show()
        
# Analze relationship between hotel occupency and availability
country_hotels = df.groupby("Country").agg({
    "Number_of_Hotels": "sum",
    "Hotel_Occupancy_Rate": "mean"})
print(country_hotels)

# Scatter plot
plt.scatter(country_hotels["Number_of_Hotels"], 
            country_hotels["Hotel_Occupancy_Rate"])
plt.xlabel("Number of Hotels")
plt.ylabel("Average Hotel Occupancy Rate (%)")
plt.title("Hotel Occupancy vs Number of Hotels by Country")

#adding country labels to the scatter plot
for i, country in enumerate(country_hotels.index):
    plt.text(country_hotels["Number_of_Hotels"][i], 
             country_hotels["Hotel_Occupancy_Rate"][i], 
             country, fontsize=8)
plt.show() 

# Relationship between tourism revenue and contribution to GDP
tourismRevenue = df.groupby("Country")[["Tourism_Revenue_USD",
"Contribution_to_GDP_Percent"]].sum()
print(tourismRevenue)

# barplot
plt.scatter(df["Tourism_Revenue_USD"], df["Contribution_to_GDP_Percent"])
plt.title("Tourism Revenue vs. Contribution to GDP")
plt.xlabel("Tourism Revenue (USD)")
plt.ylabel("Contribution to GDP (%)")
plt.show()

# correlation between tourism revenue and countribution to GDP
correlation = df["Tourism_Revenue_USD"].corr(df["Contribution_to_GDP_Percent"])
print(f"""Correlation between Tourism Revenue and Contribution to GDP: 
      {correlation}""")

# Tourist spending and contribution to GDP
plt.scatter(df["Tourist_Spending_USD"], df["Contribution_to_GDP_Percent"])
plt.title("Tourist Spending vs. Contribution to GDP")
plt.xlabel("Tourist Spending (USD)")
plt.ylabel("Contribution to GDP (%)")
plt.show()
        
# Distribution of tourist spending by purpose of visit
spendingByPurpose = df.groupby("Purpose_of_Visit")["Tourist_Spending_USD"].sum()
print(spendingByPurpose)

# barplot
spendingByPurpose.plot(kind= "bar")
plt.title("Tourist Spending by Purpose of Visit")
plt.xlabel("Purpose of Visit")
plt.ylabel("Tourist Spending (USD)")
plt.show()

# Extending with Module 4 - Cluster Analysis
removedCols = ['Hotel_Rating', 'Average_Room_Price_USD', 'Tourism_Revenue_USD', 
               'Employment_in_Tourism','Airport_Passenger_Traffic', 
               'Transport_Infrastructure_Quality', 'Carbon_Footprint_kg', 
               'Waste_Management_Rating', 'Number_of_Online_Reviews']
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

X_scaled = std_scaler.fit_transform(df_encoded[numerical_cols])

X = X_scaled

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

# utilizing Gephi to visualize the clusters
g=nx.Graph()

for n, row in df.iterrows():
    source_id = f"Country_{row['Country']}"
    target_id = f"Purpose_{row['Purpose_of_Visit']}"
    
    g.add_node(source_id, label = row["Country"], 
               cluster=str(row["Cluster"]), 
               node_type="Country")
    
    g.add_node(target_id, label = row["Purpose_of_Visit"], 
               cluster="Target Node", 
               node_type="Purpose")
    
    # 3. Add the Edge (The Trip)
    g.add_edge(source_id, 
               target_id, 
               weight=row["Tourist_Spending_USD"])

print(f"Nodes: {len(g.nodes)}")
print(f"Edges: {len(g.edges)}")

# exporting to Gephi
nx.write_graphml(g, "hp.graphml")
print("GraphML exported to: hp.graphml")