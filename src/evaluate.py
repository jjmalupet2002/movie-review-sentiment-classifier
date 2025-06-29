# evaluation and plotting functions here 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred):
    """
    Prints accuracy, precision, recall, and F1-score, and returns a classification report dict.
    """
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("F1-score:", f1_score(y_true, y_pred, average='weighted'))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    return classification_report(y_true, y_pred, output_dict=True)

def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix', save_path=None):
    """
    Plots a confusion matrix using matplotlib. If save_path is provided, saves the plot as an image instead of showing it.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved as {save_path}")
    else:
        plt.show() 