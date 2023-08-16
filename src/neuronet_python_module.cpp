#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "MnistData.hpp"
#include "Network.hpp"
#include "NetworkTrainer.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(neuronetpy, m)
{

    py::class_<MnistData>(m, "MnistData")
        .def(py::init<>())
        .def("processMNIST", py::overload_cast<>(&MnistData::processMNIST))
        .def("processMNIST", py::overload_cast<size_t, size_t>(&MnistData::processMNIST))
        .def_readwrite("train_data", &MnistData::train_data)
        .def_readwrite("train_labels", &MnistData::train_labels)
        .def_readwrite("test_data", &MnistData::test_data)
        .def_readwrite("test_labels", &MnistData::test_labels);

    py::class_<Network>(m, "Network")
        .def(py::init<std::vector<uint16_t>>())
        .def("trainMiniBatches", py::overload_cast<uint16_t, float, MnistData &>(&Network::trainMiniBatches))
        .def("randomizeWB", &Network::randomizeWB, "Randomize the weights and biasses", py::arg("min"), py::arg("max"));

    py::class_<NetworkTrainer>(m, "NetworkTrainer")
        .def(py::init<>())
        .def("trainNet", &NetworkTrainer::trainNet, "network"_a, "nEpochs"_a, "learningRate"_a, "miniBatchSize"_a, "mnistData"_a)
        .def("evalNet", &NetworkTrainer::evalNet, "network"_a, "mnistData"_a);
}