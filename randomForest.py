import random
import pandas as pd
import numpy as np
import csv
import copy
import re
from sklearn.ensemble import RandomForestClassifier
import multiprocessing

print("Start of data adjust")
data = pd.read_csv('./Dataset/data.csv')
row_swaps = ['acousticness',
 'danceability',
 'duration_ms',
 'energy',
 'explicit',
 'id',
 'instrumentalness',
 'key',
 'liveness',
 'loudness',
 'mode',
 'name',
 'popularity',
 'release_date',
 'speechiness',
 'tempo',
 'valence',
 'year',
 'artists']

# Here's how you reduce sample the data set.
# sample_data will contain every 100th row. This is just as an example
# Should use more than a 100th of the data
sample_data = data.iloc[::, :]
sample_data.head()
sample_data = sample_data[row_swaps]

# Splitting the data into features and song profiles
song_profiles = sample_data[['id', 'name', 'artists', 'release_date', 'year']].copy()
onlyartists = song_profiles.copy().filter('artists').values
features = (sample_data.copy()
    .drop('name', axis=1)
    .drop('id', axis=1)
    .drop('release_date', axis=1)
    .values.tolist())

def RemoveMultiArtistSongs(features):
    expr = re.compile("\'\,")
    output = []
    for row in features:
        artist_str = row[-1]
        if expr.match(artist_str) == None:
            output.append(row)
    return output

def GetOnly90sSongs(features):
    outfeatures = []
    for row in features:
        if row[-2] >= 1990.0 and row[-2] < 2000.0:
            outfeatures.append(row)
    return outfeatures

def GetOnly10ArtistsSongs(features):
    artists = set()
    out = []
    while len(artists) < 500:
        row = random.sample(features, 1)
        artists.add(row[0][-1])
    
    for row in features:
        if (row[-1] in artists):
            out.append(row)
    return out, artists

# def CalcClass(value, length):
#     output = [-1 for _ in range(length)]
    
#     i = 0
#     while(value > 0):
#         output[i] = 1 if (value % 2 == 1) else -1
#         value = value // 2
#         i += 1

#     return output

def CalcNormalizations(features):
    maxes = [0 for _ in range(len(features[0]))]
    mins = [30000 for _ in range (len(features[0]))]
    for row in features:
        for i in range(len(row) - 1):
            if (abs(row[i]) > maxes[i]):
                maxes[i] = abs(row[i])
            if (row[i] < mins[i]):
                mins[i] = row[i]
    
    ofeatures = copy.deepcopy(features)
    length = len(ofeatures[0])
    
    for row in ofeatures:
        for i in range(len(row) - 1):
            row[i] = (row[i] - mins[i])/(maxes[i] - mins[i])
    
    for row in ofeatures:
        for i in range(length - 1):
            row[i] = row[i]/maxes[i]

    return ofeatures

def GetNumClasses(ofeatures):
    classes = set()
    for row in ofeatures:
        classes.add(row[-1])
    return len(classes)

print("Removing songs with multiple artists...")
noMultipleArtists = RemoveMultiArtistSongs(features)
# print("Getting all songs from the 90's...")
# _90ssongs = GetOnly90sSongs(features)
print("Retrieving only 10 artists")
# _10AristsSongs, Artists = GetOnly10ArtistsSongs(_90ssongs)
_10ArtistsSongs, Artists = GetOnly10ArtistsSongs(noMultipleArtists)
Artists = list(Artists)
# Artists = onlyartists
print("Normalizing the data...")
normalized_data_with_classes = CalcNormalizations(_10ArtistsSongs)
# num_classes = GetNumClasses(features)
print("Sorting into training and testing sets...")
train_features = random.sample(normalized_data_with_classes, len(normalized_data_with_classes)*2//3)
test_features = []
for row in normalized_data_with_classes:
    if row not in train_features:
        test_features.append(row)

print("Form data for tensorflow")
tf_train_features = []
tf_train_labels = []
tf_test_features = []
tf_test_labels = []

for row in train_features:
    tf_train_features.append(row[:-1])
    tf_train_labels.append(Artists.index(row[-1]))

for row in test_features:
    tf_test_features.append(row[:-1])
    tf_test_labels.append(Artists.index(row[-1]))
tf_dataset = (tf_train_features, tf_train_labels, tf_test_features, tf_test_labels)
print("Complete!")

