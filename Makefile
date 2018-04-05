CC=g++
CFLAGS=-std=c++14

ForwardType="float"
BackwardType="float"
WeightType="float"

make:
	$(CC) $(CFLAGS) cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN -I . -Wall -Wextra -DCNN_FTYPE=$(ForwardType) -DCNN_BTYPE=$(BackwardType) -DCNN_WTYPE=$(WeightType)

test:
	$(CC) $(CFLAGS) tests/*.cpp -O3 -o TypeCNN_tests -I . -Wall -Wextra /usr/lib/libgtest.a /usr/lib/libgtest_main.a -lpthread