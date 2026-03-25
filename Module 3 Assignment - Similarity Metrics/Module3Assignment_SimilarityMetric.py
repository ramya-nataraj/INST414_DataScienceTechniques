import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import DistanceMetric

# original csv file provided by Kaggle
youtube = "most_subscribed_youtube_channels.csv"

df = pd.read_csv(youtube)

print(df.head())

# filtered to remove topics from the Youtubers list
bad_names = [
    "Music", "Gaming", "Sports", "News", 
    "YouTube Movies", "Popular on YouTube", 
    "Live", "TV Shows"
]

df = df[~df['Youtuber'].isin(bad_names)]

df = df.reset_index(drop=True)

print("Confirm filtering: \n", df.head())

# converts strings to floats
cols = ['subscribers', 'video views', 'video count']
for col in cols:
    df[col] = df[col].str.replace(',', '')  
    df[col] = df[col].astype(float)      

# select features from the dataset
features = df[['subscribers', 'video views', 'video count', 'category', 
               'started']]

# One-hot encode categories because they are currently categorical data
features = pd.get_dummies(features, columns=['category'])

# Apply L1 normalization
features_l1 = normalize(features, norm='l1')

# Convert back to DataFrame
features_l1 = pd.DataFrame(features_l1, 
                           columns=features.columns, 
                           index=features.index)

# apply Euclidean Similarity Metrics
dist = DistanceMetric.get_metric("euclidean")

print("\nComparing to T-Series: ")
# compares to T-Series
target_channel_id = df[df['Youtuber'] == 'T-Series'].index[0]

#Gathering the genres for that channel
target_channel_ratings = features_l1.loc[target_channel_id]

# Reshape correctly
target_vector = target_channel_ratings.values.reshape(1, -1)

#Generating distances from that channel to all the others
distances = dist.pairwise(features_l1, target_vector)[:,0]

query_distances = list(zip(features_l1.index, distances))

top10 = sorted(query_distances, key=lambda x: x[1], reverse=False)[:10]

#Printing the top ten most similar channel to our target
for similar_channel_id, dist_score in top10:
    print(similar_channel_id, df.loc[similar_channel_id, 'Youtuber'], dist_score)


print("\nComparing to MrBeast: ")

# compares to MrBeast
target_channel_id = df[df['Youtuber'] == 'MrBeast'].index[0]

#Gathering the genres for that channel
target_channel_ratings = features_l1.loc[target_channel_id]

# Reshape correctly
target_vector = target_channel_ratings.values.reshape(1, -1)

#Generating distances from that channel to all the others
distances = dist.pairwise(features_l1, target_vector)[:,0]

query_distances = list(zip(features_l1.index, distances))

top10 = sorted(query_distances, key=lambda x: x[1], reverse=False)[:10]

#Printing the top ten most similar channel to our target
for similar_channel_id, dist_score in top10:
    print(similar_channel_id, df.loc[similar_channel_id, 'Youtuber'], dist_score)

print("\nComparing to Cocomelon: ")

# compares to MrBeast
target_channel_id = df[df['Youtuber'] == 'Cocomelon - Nursery Rhymes'].index[0]

#Gathering the genres for that channel
target_channel_ratings = features_l1.loc[target_channel_id]

# Reshape correctly
target_vector = target_channel_ratings.values.reshape(1, -1)

#Generating distances from that channel to all the others
distances = dist.pairwise(features_l1, target_vector)[:,0]

query_distances = list(zip(features_l1.index, distances))

top10 = sorted(query_distances, key=lambda x: x[1], reverse=False)[:10]

#Printing the top ten most similar channel to our target
for similar_channel_id, dist_score in top10:
    print(similar_channel_id, df.loc[similar_channel_id, 'Youtuber'], dist_score)