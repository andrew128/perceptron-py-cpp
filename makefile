CXX = clang++
CXXFLAGS = -Wall -std=c++14
INC = -I/usr/local/include/eigen3

mll: perceptron.o
	$(CXX) $(CXXFLAGS) -O3 -g -shared -undefined dynamic_lookup `python3 -m pybind11 --includes` ml_lib.cpp -o ml_lib`python3-config --extension-suffix` $(INC) perceptron.o

perceptron.o: perceptron.cpp
	$(CXX) perceptron.cpp -O3 -g -c $(CXXFLAGS) -o perceptron.o $(INC)

clean:
	rm -rf *.dSYM *.so *.o