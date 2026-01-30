train_df, test_df = train_test_split(stored_data.processed_data, test_size=0.2, random_state=42)

#here, we find the median value of our rating, this will be the threshold that we will utilize to determine
#if a movie is a "should watch" or a "shoudnt watch"
dynamic_threshold = train_df['rating'].median()

#Again, it is important to note here that we are simply using the dynamic threshold of the train data for 
# both the test and the train data, this is because the threshold needs to be set to the median of train not test
# Also these lines just set it to 1 or 0 depending if they should watch it or not
y_train_full = (train_df['rating'] > dynamic_threshold).astype(int)
y_test       = (test_df['rating'] > dynamic_threshold).astype(int)

#Next, we drop the columns rating and vote average from our test and train input data
#We drop rating because its the value we want to predict (duh) and we also drop the vote_average because
#it is a extremely lazy predictor and we rather do it based on the embeddings, actors etc...
cols_to_drop = ['rating', 'vote_average']
X_train_full = train_df.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=['number'])
X_test       = test_df.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=['number'])