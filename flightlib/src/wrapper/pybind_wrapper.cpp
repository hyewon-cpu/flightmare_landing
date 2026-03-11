
// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// flightlib
#include "flightlib/envs/env_base.hpp"
#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"
#include "flightlib/envs/quadrotor_env/quadrotor_vis_env.hpp"
#include "flightlib/envs/quadrotor_env/quadrotor_dot_env.hpp"
#include "flightlib/envs/quadrotor_env/quadrotor_pos_env.hpp"
#include "flightlib/envs/test_env.hpp"
#include "flightlib/envs/vec_env.hpp"

namespace py = pybind11;
using namespace flightlib;

PYBIND11_MODULE(flightgym, m) {
  py::class_<VecEnv<QuadrotorEnv>>(m, "QuadrotorEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset", &VecEnv<QuadrotorEnv>::reset)
    .def("step", &VecEnv<QuadrotorEnv>::step)
    .def("testStep", &VecEnv<QuadrotorEnv>::testStep)
    .def("setSeed", &VecEnv<QuadrotorEnv>::setSeed)
    .def("close", &VecEnv<QuadrotorEnv>::close)
    .def("isTerminalState", &VecEnv<QuadrotorEnv>::isTerminalState)
    .def("curriculumUpdate", &VecEnv<QuadrotorEnv>::curriculumUpdate)
    .def("connectUnity", &VecEnv<QuadrotorEnv>::connectUnity)
    .def("disconnectUnity", &VecEnv<QuadrotorEnv>::disconnectUnity)
    .def("getNumOfEnvs", &VecEnv<QuadrotorEnv>::getNumOfEnvs)
    .def("getObsDim", &VecEnv<QuadrotorEnv>::getObsDim)
    .def("getActDim", &VecEnv<QuadrotorEnv>::getActDim)
    .def("getExtraInfoNames", &VecEnv<QuadrotorEnv>::getExtraInfoNames)
    .def("setTruncationEnabled", &VecEnv<QuadrotorEnv>::setTruncationEnabled, // hj added
         "Enable or disable episode truncation (useful for testing)")
    .def("getTruncationEnabled", &VecEnv<QuadrotorEnv>::getTruncationEnabled, // hj added
         "Get whether truncation is enabled")
    .def("__repr__", [](const VecEnv<QuadrotorEnv>& a) {
      return "RPG Drone Racing Environment";
    });

  py::class_<TestEnv<QuadrotorEnv>>(m, "TestEnv_v0")
    .def(py::init<>())
    .def("reset", &TestEnv<QuadrotorEnv>::reset)
    .def("__repr__", [](const TestEnv<QuadrotorEnv>& a) { return "Test Env"; });

  py::class_<VecEnv<QuadrotorVisEnv>>(m, "QuadrotorVisEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset", &VecEnv<QuadrotorVisEnv>::reset)
    .def("step", &VecEnv<QuadrotorVisEnv>::step)
    .def("testStep", &VecEnv<QuadrotorVisEnv>::testStep)
    .def("setSeed", &VecEnv<QuadrotorVisEnv>::setSeed)
    .def("close", &VecEnv<QuadrotorVisEnv>::close)
    .def("isTerminalState", &VecEnv<QuadrotorVisEnv>::isTerminalState)
    .def("curriculumUpdate", &VecEnv<QuadrotorVisEnv>::curriculumUpdate)
    .def("connectUnity", &VecEnv<QuadrotorVisEnv>::connectUnity)
    .def("disconnectUnity", &VecEnv<QuadrotorVisEnv>::disconnectUnity)
    .def("getNumOfEnvs", &VecEnv<QuadrotorVisEnv>::getNumOfEnvs)
    .def("getObsDim", &VecEnv<QuadrotorVisEnv>::getObsDim)
    .def("getActDim", &VecEnv<QuadrotorVisEnv>::getActDim)
    .def("getExtraInfoNames", &VecEnv<QuadrotorVisEnv>::getExtraInfoNames)
    .def("setTruncationEnabled", &VecEnv<QuadrotorVisEnv>::setTruncationEnabled,
         "Enable or disable episode truncation (useful for testing)")
    .def("getTruncationEnabled", &VecEnv<QuadrotorVisEnv>::getTruncationEnabled,
         "Get whether truncation is enabled")
    .def("__repr__", [](const VecEnv<QuadrotorVisEnv>& a) {
      return "RPG Drone Racing Visual Environment";
    });

  py::class_<VecEnv<QuadrotorDotEnv>>(m, "QuadrotorDotEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset", &VecEnv<QuadrotorDotEnv>::reset)
    .def("step", &VecEnv<QuadrotorDotEnv>::step)
    .def("testStep", &VecEnv<QuadrotorDotEnv>::testStep)
    .def("setSeed", &VecEnv<QuadrotorDotEnv>::setSeed)
    .def("close", &VecEnv<QuadrotorDotEnv>::close)
    .def("isTerminalState", &VecEnv<QuadrotorDotEnv>::isTerminalState)
    .def("curriculumUpdate", &VecEnv<QuadrotorDotEnv>::curriculumUpdate)
    .def("connectUnity", &VecEnv<QuadrotorDotEnv>::connectUnity)
    .def("disconnectUnity", &VecEnv<QuadrotorDotEnv>::disconnectUnity)
    .def("getNumOfEnvs", &VecEnv<QuadrotorDotEnv>::getNumOfEnvs)
    .def("getObsDim", &VecEnv<QuadrotorDotEnv>::getObsDim)
    .def("getActDim", &VecEnv<QuadrotorDotEnv>::getActDim)
    .def("getExtraInfoNames", &VecEnv<QuadrotorDotEnv>::getExtraInfoNames)
    .def("setTruncationEnabled", &VecEnv<QuadrotorDotEnv>::setTruncationEnabled,
         "Enable or disable episode truncation (useful for testing)")
    .def("getTruncationEnabled", &VecEnv<QuadrotorDotEnv>::getTruncationEnabled,
         "Get whether truncation is enabled")
    .def("__repr__", [](const VecEnv<QuadrotorDotEnv>& a) {
      return "RPG Drone Racing Dot Environment";
    });


    py::class_<VecEnv<QuadrotorPosEnv>>(m, "QuadrotorPosEnv_v1")
    .def(py::init<>())
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, const bool>())
    .def("reset", &VecEnv<QuadrotorPosEnv>::reset)
    .def("step", &VecEnv<QuadrotorPosEnv>::step)
    .def("testStep", &VecEnv<QuadrotorPosEnv>::testStep)
    .def("setSeed", &VecEnv<QuadrotorPosEnv>::setSeed)
    .def("close", &VecEnv<QuadrotorPosEnv>::close)
    .def("isTerminalState", &VecEnv<QuadrotorPosEnv>::isTerminalState)
    .def("curriculumUpdate", &VecEnv<QuadrotorPosEnv>::curriculumUpdate)
    .def("connectUnity", &VecEnv<QuadrotorPosEnv>::connectUnity)
    .def("disconnectUnity", &VecEnv<QuadrotorPosEnv>::disconnectUnity)
    .def("getNumOfEnvs", &VecEnv<QuadrotorPosEnv>::getNumOfEnvs)
    .def("getObsDim", &VecEnv<QuadrotorPosEnv>::getObsDim)
    .def("getActDim", &VecEnv<QuadrotorPosEnv>::getActDim)
    .def("getExtraInfoNames", &VecEnv<QuadrotorPosEnv>::getExtraInfoNames)
    .def("setTruncationEnabled", &VecEnv<QuadrotorPosEnv>::setTruncationEnabled,
         "Enable or disable episode truncation (useful for testing)")
    .def("getTruncationEnabled", &VecEnv<QuadrotorPosEnv>::getTruncationEnabled,
         "Get whether truncation is enabled")
    .def("__repr__", [](const VecEnv<QuadrotorPosEnv>& a) {
      return "RPG Drone Racing Position Environment";
    });
}
