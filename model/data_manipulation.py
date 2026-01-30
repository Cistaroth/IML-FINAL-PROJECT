import pandas as pd
import ast
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


class Model_class():
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

        self.y_dict = {"y_train_full":None, "y_test":None}
        self.X_dict = {"X_train_full":None, "X_test":None}

        self.hp_dict = {"best_k":None, 
                                          "k_values":None,
                                          "mean_balanced_accuracy_scores":None,
                                          "best_score":None }
        
        self.final_pipeline = None
        self.final_score = None

        self.y_pred = None

        self.y_prob = None

    def split_data(self):
        """
        This model will spit the data into observations (y) and predictions (X) for both test and training data sets
        
        :param self: Just self reference
        """
        train_df, test_df = train_test_split(self.processed_data, test_size=0.2, random_state=42)

        #here, we find the median value of our rating, this will be the threshold that we will utilize to determine
        #if a movie is a "should watch" or a "shoudnt watch"
        dynamic_threshold = train_df['rating'].median()

        #Again, it is important to note here that we are simply using the dynamic threshold of the train data for 
        # both the test and the train data, this is because the threshold needs to be set to the median of train not test
        # Also these lines just set it to 1 or 0 depending if they should watch it or not

        self.y_dict["y_train_full"] = (train_df['rating'] > dynamic_threshold).astype(int)
        self.y_dict["y_test"] = (test_df['rating'] > dynamic_threshold).astype(int)

        #Next, we drop the columns rating and vote average from our test and train input data
        #We drop rating because its the value we want to predict (duh) and we also drop the vote_average because
        #it is a extremely lazy predictor and we rather do it based on the embeddings, actors etc...

        cols_to_drop = ['rating', 'vote_average']
        self.X_dict["X_train_full"] = train_df.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=['number'])
        self.X_dict["X_test"] = test_df.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=['number'])

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

    def ohe_categorical_nonlist(self, columns_to_ohe: List[str] = ["language","media_type"]) -> None:
        """
        ohe categorical values that werent in lists
        
        :param columns_to_ohe: (List[str]) simply a list of the categorical values that we need to ohe
        :return: (None)
        """
        self.processed_data = pd.get_dummies(self.processed_data, columns=columns_to_ohe,dtype=int)

    def hyperparameter_search(self, max_val_k=50):
        '''
        Method that searches all wanted values of K until the optimal one is found using the balanced accuracy metric
        
        :param max_val_k: This is the maximum value of K that we are going to search in
        '''
        self.hp_dict["k_values"] =  range(1, max_val_k)
        self.hp_dict["mean_balanced_accuracy_scores"] =  []

        print("Starting Hyperparameter Search...")

        #For every K we:
        for k in self.hp_dict["k_values"]:
            #Create a pipeline
            pipeline = make_pipeline(
                SimpleImputer(strategy='median'), 
                MinMaxScaler(), 
                KNeighborsClassifier(n_neighbors=k)
            )
            
            #Create a stratifiedKfold cross validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # We pass the X_train_full we created manually above using balanced accuracy as our scoring metric
            scores = cross_val_score(pipeline, self.X_dict["X_train_full"], self.y_dict["y_train_full"], cv=cv, scoring='balanced_accuracy')
            
            #We calculate the mean of the scores and add it to our list
            self.hp_dict["mean_balanced_accuracy_scores"].append(scores.mean())

        #Now, we find the best score in our mean balanced accuracy scores
        best_score_index = np.argmax(self.hp_dict["mean_balanced_accuracy_scores"])

        self.hp_dict["best_k"] = self.hp_dict["k_values"][best_score_index]
        self.hp_dict["best_score"] = self.hp_dict["mean_balanced_accuracy_scores"][best_score_index]

        #We get our best K value and score value
        print(f"The Optimal K is: {self.hp_dict["best_k"]} with a Validation Accuracy of {self.hp_dict["best_score"]:.2%}")

    def create_final_pipeline(self):
        #best_k,k_values,mean_balanced_accuracy_scores, best_score = hyperparameter_search()


        #After that, we simply create a pipeline with the best k value
        self.final_pipeline = make_pipeline(
                    SimpleImputer(strategy='median'), 
                    MinMaxScaler(), 
                    KNeighborsClassifier(n_neighbors=self.hp_dict["best_k"]))

        #We fit it with ALL of the train data
        self.final_pipeline.fit(self.X_dict["X_train_full"], self.y_dict["y_train_full"]) 

        #And obtain our final score
        self.final_score = self.final_pipeline.score(self.X_dict["X_test"], self.y_dict["y_test"])
        print(f"Final Test Accuracy: {self.final_score:.2f}")

    def calculate_final_prediction(self):
        self.y_pred = self.final_pipeline.predict(self.X_dict["X_test"])
        self.y_prob = self.final_pipeline.predict_proba(self.X_dict["X_test"])[:, 1]

    def output_non_dim_columns_with_example_data(self) -> None:

        """
        Helper function to just see the columns that arent embeddings and a example value of them
        mainly used for debugging
        
        :return: (None)
        """
        [print("Name:", column, "Val:",self.processed_data[column][0]) for column in self.processed_data if "dim" not in str(column)]
