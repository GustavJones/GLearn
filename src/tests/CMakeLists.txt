cmake_minimum_required(VERSION 3.15)
project(GLearn-Tests CXX)

file(GLOB SOURCES "*.cpp")
add_executable(tests ${SOURCES})
target_include_directories(tests PRIVATE ../../include)
target_link_libraries(tests PRIVATE GLearn-NeuralNetwork GLearn-Data)
