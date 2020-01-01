#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "perceptron.h"

namespace py = pybind11;

PYBIND11_MODULE(ml_lib, m) {
    m.doc() = "Machine Learning Library";

    py::class_<Perceptron>(m, "Perceptron")
        .def(py::init<int, float, int>())
        .def("train", &Perceptron::train)
        .def("get_weights", &Perceptron::get_weights);
}