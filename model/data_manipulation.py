"""
This is going to be the file where we first take our data from the processed data, we are going to modify it and make it
be in the correct format so that our KNN model can do its job. For now the things that this file has to do are:

1. Get data from processed_data from userdata_embeddings.csv
2. One hot encode the data for the final few columns, specifically ,"language","media_type","vote_average","production_countries","genres","credits"
4. Scale the values from the embeddings (normalisation)
5. Combine everything into a huge matrix
6. Finally feed these vectors into our KNN to find the closet movie to the one that we wanted.
"""

import pandas as pd

#Reading the user data
df = pd.read_csv("processed_data/userdata_embeddings.csv")

encoded_df = pd.get_dummies(df, columns=["language","media_type","vote_average","production_countries","genres","credits","runtime","popularity","rating"])
encoded_df = encoded_df.astype(str)

testing = df[["language","media_type","vote_average","production_countries","genres","credits","runtime","popularity","rating"]]

print(testing.head())
