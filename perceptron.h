#include <iostream>
#include <Eigen/Dense>
#include <vector>

class Perceptron {
    float learning_rate;
    int num_epochs;
    Eigen::VectorXf W;

public:
    Perceptron(int input_size, float input_learning_rate, int input_num_epochs) : 
      learning_rate(input_learning_rate), num_epochs(input_num_epochs) {
        W = Eigen::VectorXf::Zero(input_size + 1);
    }

    int step_activation(int x);

    int predict(Eigen::VectorXf &x);

    std::vector<float> train(Eigen::MatrixXf &X, Eigen::VectorXf &y);

    Eigen::VectorXf get_weights(Eigen::MatrixXf &X, Eigen::VectorXf &y);

    float get_accuracy(Eigen::MatrixXf &X, Eigen::VectorXf &y);
};