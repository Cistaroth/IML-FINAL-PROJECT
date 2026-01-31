from model.model_class import KNN_movie_recomender
from model.evaluation import (
    plot_confusion_matrix,
    plot_hyperparameter_search,
    plot_roc_curve,
    get_detailed_metrics,
)


def main():
    # We create an object of type Data that will hold the raw and processed data.
    recomender_instance = KNN_movie_recomender()

    # We one hot encode the data before splitting it
    # We know this can add a small amount of leakage in columns like "credits",
    # However, this has been done to avoid having clashes with columns not existing in our training that do in our test
    # Its a small price to pay for simplicity

    recomender_instance.ohe_lists()
    recomender_instance.ohe_categorical_nonlist()

    # Now, we make the split between test and train data, we do a 80% train, 20% test split

    recomender_instance.split_data()

    # Now, after we have curated the data into their train and test configurations, we need to find the ideal K value
    # We set our range of possible K values from 1 to max_val_k

    recomender_instance.hyperparameter_search(max_val_k=50)

    # Finally after finding our optimal k value, we create the final pipeline
    recomender_instance.create_final_pipeline()

    # Then, we calculate our prediction
    recomender_instance.calculate_final_prediction()

    # We output the metrics of our model
    get_detailed_metrics(recomender_instance)

    # And we do some plots
    plot_roc_curve(recomender_instance)
    plot_hyperparameter_search(recomender_instance.hp_dict)
    plot_confusion_matrix(recomender_instance)


if __name__ == "__main__":
    main()
