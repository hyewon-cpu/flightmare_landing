# CMake generated Testfile for 
# Source directory: /home/flightmare_develop/flightlib
# Build directory: /home/flightmare_develop/flightlib/build_local
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_lib "test_lib")
set_tests_properties(test_lib PROPERTIES  _BACKTRACE_TRIPLES "/home/flightmare_develop/flightlib/CMakeLists.txt;260;add_test;/home/flightmare_develop/flightlib/CMakeLists.txt;0;")
add_test(test_unity_bridge "test_unity_bridge")
set_tests_properties(test_unity_bridge PROPERTIES  _BACKTRACE_TRIPLES "/home/flightmare_develop/flightlib/CMakeLists.txt;270;add_test;/home/flightmare_develop/flightlib/CMakeLists.txt;0;")
subdirs("externals/pybind11-src")
subdirs("/home/flightmare_develop/flightlib/externals/yaml-build")
subdirs("/home/flightmare_develop/flightlib/externals/googletest-build")
