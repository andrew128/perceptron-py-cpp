import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size + 1) # +1 for bias
        self.epochs = epochs
        self.lr = lr

    def step_activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        return self.step_activation(self.W.T.dot(x))

    def train(self, X, y):
        accuracies = []
        for epoch in range(self.epochs):
            for i in range(y.shape[0]):
                x = np.insert(X[i], 0, 1) # add bias
                preds = self.predict(x)
                error = y[i] - preds
                self.W = self.W + self.lr * error * x
            if epoch % 1000 == 0:
                accuracy = self.get_accuracy(X, y)
                accuracies.append(accuracy)
                print('Epoch:', epoch, 'Accuracy:', accuracy)

        return accuracies


    def get_accuracy(self, X, y):
        num_correct = 0
        for i in range(y.shape[0]):
            x = np.insert(X[i], 0, 1) # add bias
            preds = self.predict(x)
            error = y[i] - preds
            if error == 0:
                num_correct += 1

        return num_correct / y.shape[0]

def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    perceptron = Perceptron(input_size=2)
    perceptron.train(X, y)
    print(perceptron.W)

if __name__ == '__main__':
    main()