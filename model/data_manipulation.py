import pandas as pd
import ast
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from typing import List

class Data():
    def __init__(self, data_location: str = "processed_data/userdata_embeddings.csv", max_features_kept: int = 20) -> None:
        """
        Initialise the data of a given user

        :param data_location: (str) This is the directory where the csv file containing the data is stored
        :param max_features_kept: (int) this is the maximum number of features that our model will keep when ohe a list
        :return: (None)
        """

        self.data_location = data_location
        self.raw_data = pd.read_csv(data_location) #Could lead to data duplicated in memory, definitely something to keep in mind
        self.processed_data = deepcopy(self.raw_data)
        self.count_vectorizer = CountVectorizer(max_features=max_features_kept, stop_words=None)
        self.scaler = MinMaxScaler()
    
    def create_curated_string(self, obj: str, n_kept: int|None = 3) -> List[str]:
        '''
        Helper function to curate a list from ["Ricardo Rubert",...] to "RicardoRubert ..."
        
        :param obj: (str) This is the list that we are going to be fixing
        :param n_kept: (int) this is the number of values that we are going to keep from the top, so 3 means
                        we only keep 3 of the names.
        :return: (List[str])
        '''
        curated_list = ast.literal_eval(obj)
        if n_kept is not None:
            curated_list = curated_list[:n_kept] 
        return [list_index.replace(" ", "").lower() for list_index in curated_list]

    def ohe_lists(self, columns: List[str] = ["credits", "genres", "production_countries"]) -> None:
        """
        One hot encode lists properly, while trying to avoide curse of dimensionality

        :param columns: (List[str]) A list of the columns that contain list data that we need to ohe
        :return: (None)
        """

        vectorized_dataframes = []

        for column in columns:
            self.processed_data[column] = self.processed_data[column].apply(self.create_curated_string)
            self.processed_data[column] = self.processed_data[column].apply(lambda x: " ".join(x))

            count_matrix = self.count_vectorizer.fit_transform(self.processed_data[column])
            vectorized_dataframes.append(pd.DataFrame(count_matrix.toarray(), columns=self.count_vectorizer.get_feature_names_out()))

            self.processed_data.drop(columns=[column], inplace=True, axis=1)
        
        self.processed_data = pd.concat([self.processed_data] + vectorized_dataframes, axis=1)
    
    def normalize_continuous_variables(self, columns_to_scale: List[str] = ["year", "vote_average", "runtime", "popularity", "rating"]) -> None:
        """
        Normalise continous variables between 1 and 0 utilizing MinMaxScaler by default
        
        :param columns_to_scale: (List[str]) A list of the continuous data columns that we need to scale
        :return: (None)
        """

        self.processed_data[columns_to_scale] = self.scaler.fit_transform(self.processed_data[columns_to_scale])
    
    def ohe_categorical_nonlist(self, columns_to_ohe: List[str] = ["language","media_type"]) -> None:
        """
        ohe categorical values that werent in lists
        
        :param columns_to_ohe: (List[str]) simply a list of the categorical values that we need to ohe
        :return: (None)
        """
        self.processed_data = pd.get_dummies(self.processed_data, columns=columns_to_ohe,dtype=int)
    
    def output_non_dim_columns_with_example_data(self) -> None:
        """
        Helper function to just see the columns that arent embeddings and a example value of them
        mainly used for debugging
        
        :return: (None)
        """
        [print("Name:", column, "Val:",self.processed_data[column][0]) for column in self.processed_data if "dim" not in str(column)]