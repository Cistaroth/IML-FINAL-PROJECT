from model.data_manipulation import Model_class
from model.evaluation import plot_confusion_matrix, plot_hyperparameter_search, plot_roc_curve, get_detailed_metrics

def main():
    #We create an object of type Data that will hold the raw and processed data.
    model = Model_class()

    #We one hot encode the data before splitting it
    # We know this can add a small amount of leakage in columns like "credits", 
    # However, this has been done to avoid having clashes with columns not existing in our training that do in our test
    # Its a small price to pay for simplicity

    model.ohe_lists()
    model.ohe_categorical_nonlist()

    #Now, we make the split between test and train data, we do a 80% train, 20% test split

    model.split_data()

    #Now, after we have curated the data into their train and test configurations, we need to find the ideal K value
    #We set our range of possible K values from 1 to max_val_k

    model.hyperparameter_search(max_val_k=50)

    model.create_final_pipeline()

    model.calculate_final_prediction()

    get_detailed_metrics(model)

    plot_roc_curve(model)
    plot_hyperparameter_search(model.hp_dict)
    plot_confusion_matrix(model)


if __name__ == "__main__":
    main()
