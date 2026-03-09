import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

harry_potter = "merged_relations.csv"

df = pd.read_csv(harry_potter)


# first five rows
# print(df.head())

# Get the column names as a list
column_names = df.columns.tolist()
print(column_names)

# frequency values of relationships
hp_freq = df["type"].value_counts()

# barplot
plt.bar(hp_freq.index, hp_freq.values)
plt.title("Frequency of Harry Potter Relations")
plt.xlabel("Relationship Type")
plt.ylabel("Frequency")
plt.show()

# Top ten relationships
counts = df["source_name"].value_counts().head(10)

plt.bar(counts.index, counts.values)
plt.xlabel("Character")
plt.ylabel("Number of Relationships")
plt.title("Top Characters by Relationships")
plt.show()

# Graph creation
g=nx.Graph()

# Nodes and Edges Counts
for n, row in df.iterrows():
    g.add_node(row["source"], name=row["source_name"])
    g.add_node(row["target"], name=row["target_name"])
    
    g.add_edge(row["source"], row["target"], type=row["type"])
    

print("Nodes:",len(g.nodes))
print("Edges:", len(g.edges))    

# graph used for Gephi
nx.write_graphml(g, "hp.graphml")
print("GraphML exported to: hp.graphml")

top_k = 20
deg_list=sorted(g.degree, key=lambda x:x[1], reverse= True)[:top_k]

print("\n Top 20 characters by number of neighbors:")
for node_id, deg in deg_list:
    print(f"{g.nodes[node_id].get('name', node_id)} - {deg}")



