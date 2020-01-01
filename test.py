import ml_lib
import numpy as np
from perceptron import Perceptron

def get_data(path='./data.txt'):
    '''
    :return: list of tuples (x, y, class)
    '''
    output = []
    with open(path, 'r') as f:
	    file_data = f.read().split()

    for line in file_data:
        output.append(line.split(','))

    data = np.asarray(output).astype(np.float32)
    
    X = data[:,:2]
    y = data[:,2]

    return (X, y)

def main():
    X, y = get_data()

    # C++
    p = ml_lib.Perceptron(2,1, 50000)
    cpp_accuracies = p.train(X, y)
    print(p.get_weights(X, y))

    # Python
    perceptron = Perceptron(input_size=2, epochs=num_epochs)
    py_accuracies = perceptron.train(X, y)
    print(perceptron.W)


if __name__ == '__main__':
    main()