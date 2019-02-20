import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))

from src.legacy.TeamB1pomt5.code.config import LOG_DIR, FIGURES_DIR

def loss_curves(log_file):
    '''
    Plots training and validation losses over epochs.
    '''
    filename = os.path.splitext(os.path.basename(log_file))[0]
    curve_dir = os.path.join(FIGURES_DIR, 'learning_curves')

    # Get losses
    with open(log_file, 'r') as fp:
        lines = fp.readlines()[1:]  # Strip header
    train_losses, valid_losses = [], []
    for line in lines:
        split = line.split(',')
        train_losses.append(float(split[1]))
        valid_losses.append(float(split[2]))
    epochs = range(len(lines))

    # Plot
    plt.title('{} - Loss curves'.format(filename))
    plt.plot(epochs, train_losses, color='blue', label='Training loss')
    plt.plot(epochs, valid_losses, color='red', label='Validation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(curve_dir, '{}_losses.png'.format(filename)))
    plt.gcf().clear()

def acc_curves(log_file):
    '''
    Plots training and validation accuracies over epochs.
    '''
    filename = os.path.splitext(os.path.basename(log_file))[0]
    curve_dir = os.path.join(FIGURES_DIR, 'learning_curves')

    # Get losses
    with open(log_file, 'r') as fp:
        lines = fp.readlines()[1:]  # Strip header
    train_accs, valid_accs = [], []
    for line in lines:
        split = line.split(',')
        train_accs.append(float(split[3])*100)
        valid_accs.append(float(split[4])*100)
    epochs = range(len(lines))

    # Plot
    plt.title('{} - Accuracy curves'.format(filename))
    plt.plot(epochs, train_accs, color='blue', label='Training accuracy')
    plt.plot(epochs, valid_accs, color='red', label='Validation accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(curve_dir, '{}_accs.png'.format(filename)))
    plt.gcf().clear()

if __name__ == '__main__':
    for file in os.listdir(LOG_DIR):
        if file.endswith('.losses'):
            loss_curves(os.path.join(LOG_DIR, file))
            #acc_curves(os.path.join(LOG_DIR, file))