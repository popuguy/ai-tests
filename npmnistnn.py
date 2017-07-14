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

display_mnist(mnist.train.images[0], str(mnist.train.labels[0].nonzero()[0]))

hidden_layer_1_num_nodes = 500
hidden_layer_2_num_nodes = 500
hidden_layer_3_num_nodes = 500
output_layer_num_nodes = 10
batch_size = 100
dimension = 28
full_iterations = 10

def convert_digit_to_onehot(digit):
    return [0] * digit + [1] + [0] * (9 - digit)

# def load_images():
#     images = get_image_strings()
#     labels = get_image_labels()
#     dic = {}
#     for i in range(len(images)):
#         dic[images[i]] = labels[i]
#     return dic
#
# all_images = load_images()

#all must be numerical
# images = np.array([[0] * dimension**2]] * NUMBER_OF_IMAGES)
# labels = np.array([[1]] * NUMBER_OF_IMAGES)
images = mnist.train.images
labels = mnist.train.labels
test_images = np.array()
test_labels = np.array()
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def slope_from_sigmoid(x):
    return x * (1 - x)

syn1 = 2 * np.random.random((dimension**2, hidden_layer_1_num_nodes)) - 1
syn2 = 2 * np.random.random((hidden_layer_1_num_nodes, hidden_layer_2_num_nodes)) - 1
syn3 = 2 * np.random.random((hidden_layer_2_num_nodes, hidden_layer_3_num_nodes)) - 1
syn4 = 2 * np.random.random((hidden_layer_3_num_nodes, output_layer_num_nodes)) - 1
for iter in range(full_iterations):
    for section in range(0, len(images), batch_size):
        training_images = images[section:section+batch_size]
        training_labels = labels[section:section+batch_size]
        # l0 = flatten_image(inputimageshitpixels)
        l0 = training_images
        # l1 = sigmoid(l0 DOT layer1synapses)
        l1 = sigmoid(np.dot(l0, syn1))
        # l2 = sigmoid(l1 DOT layer2synapses)
        l2 = sigmoid(np.dot(l1, syn2))
        # l3 = sigmoid(l2 DOT layer3synapses)
        l3 = sigmoid(np.dot(l2, syn3))
        # l4 = sigmoid(l3 DOT layer4synapses)
        l4 = sigmoid(np.dot(l3, syn4))
        # l4_err = l4 - intended output aka (0,0,1,0,0,0,0,0,0,0) would mean 100% the number 2 since each place is the corresponding digit
        l4_err = l4 - training_labels
        # l4_delta = l4_err * slope_on_sigmoid_curve(l4)
        l4_delta = l4_err * slope_from_sigmoid(l4)
        # l3_err = l4_delta DOT layer4synapses transposition (basically find the contribution of error shit from the weights with correct lin alg)
        l3_err = np.dot(l4_delta, syn4.T)
        # l3_delta = l3_err * slope_on_sigmoid_curve(l3)
        l3_delta = l3_err * slope_from_sigmoid(l3)
        # l2_err = l3_delta DOT layer3synapses transposition (backpropagate but prob maybe do this shit with loops or sumfin irl)
        l2_err = np.dot(l3_delta, syn3.T)
        # l2_delta = l2_err * slope_on_sigmoid_curve(l2)
        l2_delta = l2_err * slope_from_sigmoid(l2)
        # l1_err = l2_delta DOT layer2synapses transposition
        l1_err = np.dot(l2_delta, syn2.T)
        # l1_delta = l1_err * slope_on_sigmoid_curve(l1)
        l1_delta = l1_err * slope_from_sigmoid(l1)
        # layer4synapses += l3 transposition DOT l4_delta
        # layer3synapses += l2 transposition DOT l3_delta
        # layer2synapses += l1 transposition DOT l2_delta
        # layer1synapses += l0 transposition DOT l1_delta
        syn4 += np.dot(l3.T, l4_delta)
        syn3 += np.dot(l2.T, l3_delta)
        syn2 += np.dot(l1.T, l2_delta)
        syn1 += np.dot(l0.T, l1_delta)
        
