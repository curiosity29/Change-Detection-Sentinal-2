import matplotlib.pyplot as plt
import numpy as np

def plot_prediction(image, label, prediction, channel_first = True, argmax = False):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    if channel_first:
        image= np.transpose(image, (1,2,0))
        label= np.transpose(label, (1,2,0))
        prediction= np.transpose(prediction, (1,2,0))

    ax = fig.add_subplot(1, 4, 1, xticks=[], yticks=[])
    # n_channel = int(min(image.shape[-1], 3))
    n_class = int(min(label.shape[-1], 3))
    
    plt.imshow(image[..., :3])
    ax.set_title("image 1")
    ax = fig.add_subplot(1, 4, 2, xticks=[], yticks=[])
    plt.imshow(image[..., 4:7])
    ax.set_title("image 2")
    ax = fig.add_subplot(1, 4, 3, xticks=[], yticks=[])
    plt.imshow(label[..., :n_class])
    ax.set_title("label")
    ax = fig.add_subplot(1, 4, 4, xticks=[], yticks=[])
    plt.imshow(prediction[..., :n_class])
    ax.set_title("prediction")
    plt.tight_layout()
    return fig
