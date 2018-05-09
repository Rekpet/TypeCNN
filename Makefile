CC=g++
CFLAGS=-std=c++14

.PHONY: typecnn fixed tests

make: typecnn

all: typecnn fixed tests

# Default values for types are floats, can be omitted

typecnn:
	$(CC) $(CFLAGS) cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN -I . -Wall -Wextra -DCNN_FTYPE=float -DCNN_BTYPE=float -DCNN_WTYPE=float

fixed:
	$(CC) $(CFLAGS) cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_fixed -I . -Wall -Wextra -DCNN_FTYPE="FixedPoint<8,8>" -DCNN_BTYPE="float" -DCNN_WTYPE="FixedPoint<8,8>"

tests:
	$(CC) $(CFLAGS) tests/*.cpp -O3 -o TypeCNN_tests -I . -Wall -Wextra /usr/lib/libgtest.a /usr/lib/libgtest_main.a -lpthread -DCNN_FTYPE=float -DCNN_BTYPE=float -DCNN_WTYPE=float
