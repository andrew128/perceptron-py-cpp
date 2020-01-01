#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "perceptron.h"

int Perceptron::step_activation(int x) {
    if (x >= 0) return 1;
    else return 0;
}

int Perceptron::predict(Eigen::VectorXf &x) {
    int z = (W).dot(x);
    return this->step_activation(z);
}

std::vector<float> Perceptron::train(Eigen::MatrixXf &X, Eigen::VectorXf &y) {
    std::vector<float> accuracies;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (int i = 0; i < (y).size(); i++) {
            Eigen::VectorXf x_with_bias(X.row(i).size() + 1);
            x_with_bias << 1, X.row(i).transpose();
            int pred = this->predict(x_with_bias);
            auto error = y[i] - pred;
            W = W + learning_rate * error * x_with_bias;
        }
        if (epoch % 1000 == 0) {
            float accuracy = this->get_accuracy(X, y);
            accuracies.push_back(accuracy);
            std::cout << "Epoch: " << epoch << " Accuracy: " << accuracy << "\n";
        }
    }

    return accuracies;
}

Eigen::VectorXf Perceptron::get_weights(Eigen::MatrixXf &X, Eigen::VectorXf &y) {
    return this->W;
}

float Perceptron::get_accuracy(Eigen::MatrixXf &X, Eigen::VectorXf &y) {
    float num_correct = 0;
    for (int i = 0; i < (y).size(); i++) {
        Eigen::VectorXf x_with_bias(X.row(i).size() + 1);
        x_with_bias << 1, X.row(i).transpose();
        int pred = this->predict(x_with_bias);
        auto error = y[i] - pred;
        if (error == 0) num_correct += 1;
    }
    return num_correct / y.size();
}

int main() {}