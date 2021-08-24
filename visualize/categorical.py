
import torch

def plot_confusion(n_class, ans, pred, filename="confusuion.png"):
    confusion = {label: {pred:0 for pred in range(n_class)} for label in range(n_class)}

    for l, p in zip(ans, pred):
        confusion[l.item()][torch.argmax(p).item()] += 1

    print(f"Confusion matrix of: {filename}")
    for latent in range(n_class):
        print(confusion[latent])
    return