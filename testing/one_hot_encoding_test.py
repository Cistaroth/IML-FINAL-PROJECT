"""
This file will be to test how to one hot encode something using pandas.
"""


import pandas as pd


data = {
    'Title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
    'Genre': ['Comedy', 'Horror', 'Comedy', 'Action']
}

df = pd.DataFrame(data)

print("---Original Data---")
print(df)

encoded_df = pd.get_dummies(df, columns=['Genre'])
encoded_df = encoded_df.astype(str)

print("\n--- One-Hot Encoded Data ---")
print(encoded_df)