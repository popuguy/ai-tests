import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def display_mnist(img, label):
    '''Visually display the 28x28 unformatted array
    '''
    basic_array = img
    plt.imshow(basic_array.reshape((28,28)), cmap=cm.Greys)
    plt.suptitle('Image is of a ' + label)
    plt.show()

hidden_layer_1_num_nodes = 500
hidden_layer_2_num_nodes = 500
hidden_layer_3_num_nodes = 500
output_layer_num_nodes = 10
batch_size = 100
dimension = 28
full_iterations = 10

def convert_digit_to_onehot(digit):
    return [0] * digit + [1] + [0] * (9 - digit)

images = mnist.train.images
labels = mnist.train.labels
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def slope_from_sigmoid(x):
    return x * (1 - x)

syn1 = 2 * np.random.random((dimension**2, hidden_layer_1_num_nodes)) - 1
syn2 = 2 * np.random.random((hidden_layer_1_num_nodes, hidden_layer_2_num_nodes)) - 1
syn3 = 2 * np.random.random((hidden_layer_2_num_nodes, hidden_layer_3_num_nodes)) - 1
syn4 = 2 * np.random.random((hidden_layer_3_num_nodes, output_layer_num_nodes)) - 1
for iter in range(full_iterations):
    print('Epic epoch bro, we\'re at #' + str(iter+1))
    for section in range(0, len(images), batch_size):
        training_images = images[section:section+batch_size]
        training_labels = labels[section:section+batch_size]
        l0 = training_images
        l1 = sigmoid(np.dot(l0, syn1))
        l2 = sigmoid(np.dot(l1, syn2))
        l3 = sigmoid(np.dot(l2, syn3))
        l4 = sigmoid(np.dot(l3, syn4))
        l4_err = training_labels - l4
        l4_delta = l4_err * slope_from_sigmoid(l4)
        l3_err = np.dot(l4_delta, syn4.T)
        l3_delta = l3_err * slope_from_sigmoid(l3)
        l2_err = np.dot(l3_delta, syn3.T)
        l2_delta = l2_err * slope_from_sigmoid(l2)
        l1_err = np.dot(l2_delta, syn2.T)
        l1_delta = l1_err * slope_from_sigmoid(l1)
        syn4 += np.dot(l3.T, l4_delta)
        syn3 += np.dot(l2.T, l3_delta)
        syn2 += np.dot(l1.T, l2_delta)
        syn1 += np.dot(l0.T, l1_delta)
correct = 0
total = 0
l4list = l4.tolist()
training_labelslist = training_labels.tolist()
print('Num things', len(l4list))
for i in range(len(l4list)):
    print(l4list[i])
    # display_mnist(l0[i], str(l4list[i].index(max(l4list[i]))))
    if l4list[i].index(max(l4list[i])) == training_labelslist[i].index(max(training_labelslist[i])):
        correct += 1
    total += 1
print('Final round', 100*(correct/total),'percent correct')
