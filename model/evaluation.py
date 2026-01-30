import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, balanced_accuracy_score, roc_auc_score, roc_curve, auc
import pandas as pd

def plot_roc_curve(model):

    fpr, tpr, _ = roc_curve(model.y_dict["y_test"], model.y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def plot_hyperparameter_search(hp_dict):

    plt.figure(figsize=(10, 6))
    plt.plot(hp_dict["k_values"], hp_dict["mean_balanced_accuracy_scores"], marker='o', linestyle='-', color='b', label='Validation Accuracy')


    plt.plot(hp_dict["best_k"], hp_dict["best_score"], marker='*', color='r', markersize=15, label=f'Best K={hp_dict["best_k"]}')

    plt.title('Hyperparameter Tuning: Balanced Accuracy vs. K Neighbors')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-Validated Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def plot_confusion_matrix(model):

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_estimator(
        model.final_pipeline, 
        model.X_dict["X_test"], 
        model.y_dict["y_test"], 
        display_labels=['Skip', 'Watch'], 
        cmap='Blues',                     
        normalize=None,
        ax=ax
    )

    plt.title("Confusion Matrix: Prediction Performance")
    plt.show()


def get_detailed_metrics(model):
    print("--- DETAILED METRICS ---")

    # A. The "Big Picture" Report (Precision, Recall, F1)
    print(classification_report(model.y_dict["y_test"], model.y_pred, target_names=['Skip', 'Watch']))

    # B. Balanced Accuracy
    bal_acc = balanced_accuracy_score(model.y_dict["y_test"], model.y_pred)
    print(f"Balanced Accuracy: {bal_acc:.2f}")

    # C. ROC-AUC Score
    # Note: This requires the probabilities (y_prob), not the hard 0/1 predictions
    try:
        roc_score = roc_auc_score(model.y_dict["y_test"], model.y_prob)
        print(f"ROC-AUC Score:     {roc_score:.2f}")
    except ValueError:
        print("ROC-AUC requires both classes to be present in test set.")

    cm = confusion_matrix(model.y_dict["y_test"], model.y_pred)

    print("\n--- Labeled Confusion Matrix ---")
    # Creates a table: Rows = Actual, Cols = Predicted
    cm_df = pd.DataFrame(cm, index=["Actual Skip", "Actual Watch"], columns=["Pred Skip", "Pred Watch"])
    print(cm_df)
