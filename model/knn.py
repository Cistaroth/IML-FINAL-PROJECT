"""
Here is where the actual model will be placed into.
"""
from sklearn.neighbors import NearestNeighbors


#the K value still has to be decided
#Cosine was chosen because we dont care about the magnitude, but we care about the direction instead
#sklearn is smart so we just let it use whatever algorithm it wants
knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='auto')

#Next we need to fit it but I havent done the data pre-processing yet so that is our next step