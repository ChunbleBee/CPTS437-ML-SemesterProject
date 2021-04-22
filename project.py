import pandas as pd
import numpy as np

data = pd.read_csv('./Dataset/data.csv')

# Here's how you reduce sample the data set. sample_data will contain every 100th row. This is just as an example
# Should use more than a 100th of the data
sample_data = data.iloc[::100, :]
sample_data.head()

# Splitting the data into features and song profiles
song_profiles = sample_data[['id', 'name', 'artists', 'release_date', 'year']].copy()
features = sample_data.drop('name', axis=1).copy()
features = features.drop('artists', axis=1)

# Drop irrelevant columns in feature set
features = features.drop('mode', axis=1)
# features = features.drop('key', axis=1) # Key?
song_profiles.head()
features.head()

if __name__ == "__main__":
    print(features)
