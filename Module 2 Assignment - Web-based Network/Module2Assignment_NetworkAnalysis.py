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

# create a stacked bar graph that shows the positive & negative relationships
# of the top ten characters
# count relationships per character
# combine all character appearances
all_chars = pd.concat([df["source_name"], df["target_name"]])

counts = all_chars.groupby(all_chars).size().sort_values(ascending=False).head(10)

pos = []
neg = []

for name in counts.keys():
    pos.append(df[((df["source_name"] == name) | (df["target_name"] == name)) & 
                  (df["type"] == "+")].shape[0])
    neg.append(df[((df["source_name"] == name) | (df["target_name"] == name)) & 
                  (df["type"] == "-")].shape[0])
    
    # verify counts
    print(name, pos, neg)

plt.bar(counts.keys(), pos, label="Positive")
plt.bar(counts.keys(), neg, bottom=pos, label="Negative")
plt.xlabel("Characters")
plt.ylabel("Relationships Counts split by Positive and Negative")
plt.title("Top Characters by Relationships Types")

plt.xticks(rotation=45)
plt.legend()
plt.show()

