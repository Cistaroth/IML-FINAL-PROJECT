"""
This file will hold the object that we will utilize for the user, here we will store the movies they like
dislike, and we will utilize these so that we can create an average of the movie they like and recomend the best ones.
"""

class User():
    def __init__(self, data_object):
        self.liked_movies = []
        self.disliked_movies = []
        self.data = data_object

    
