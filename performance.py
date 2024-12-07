import matplotlib.pyplot as plt
import seaborn as sns
import time

def training_validation(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def confusion_matrix(test_labels, predictions, actions):
    from sklearn.metrics import confusion_matrix

    # Generate confusion matrix
    conf_matrix = confusion_matrix(test_labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


accuracy_history = []
accurate_count = 0
total_count = 0

def accuracy_measurement(accuracy):
    global accurate_count, total_count, accuracy_history

    accurate_count += 1 if accuracy else 0
    total_count += 1
    
    accuracy_history.append(accurate_count / total_count * 100)


def return_accuracy_history():
    global accuracy_history
    return accuracy_history


# Plot accuracy function
def plot_accuracy():
    global accuracy_history

    # Initialize the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Accuracy', color='blue')  # Initialize an empty line plot
    ax.set_xlabel('Prediction Steps')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Continuous Accuracy Measurement')
    ax.legend(loc='lower right')

    while True:
        x_data = list(range(len(accuracy_history))) 
        y_data = accuracy_history 

        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.relim()
        ax.autoscale_view()

        plt.draw() 
        plt.pause(0.1)

        time.sleep(0.5) 