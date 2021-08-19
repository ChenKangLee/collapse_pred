import torch
import matplotlib.pyplot as plt

def plot_loss(training_loss, valid_loss, filename='loss.png'):
    fig, ax = plt.subplots() 

    ax.plot(range(len(training_loss)), training_loss, '-b', label='training')
    ax.plot(range(len(valid_loss)), valid_loss, '-g', label='validation')

    plt.xlabel("epoch")
    plt.legend(loc='upper left')
    plt.title('train/valid loss curve')

    plt.savefig(filename)
    # plt.show()


def plot_pred(ans, pred, filename="out.png"):
    fig, ax = plt.subplots() 

    ax.plot(range(len(ans)), ans, '.k', label='ans', markersize=5)
    ax.plot(range(len(pred)), pred, '.r', label='prediction', markersize=5)

    plt.xlabel("sample")
    plt.legend(loc='upper left')
    plt.title('Prediction results')

    # plt.show()
    plt.savefig(filename)


def plot_confusion(n_class, ans, pred, filename="confusuion.png"):
    confusion = {label: {pred:0 for pred in range(n_class)} for label in range(n_class)}

    for l, p in zip(ans, pred):
        confusion[l.item()][torch.argmax(p).item()] += 1

    print(f"Confusion matrix of: {filename}")
    print(confusion)
    return
