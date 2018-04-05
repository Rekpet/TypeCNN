CC=g++
CFLAGS=-std=c++14

.PHONY: all benchmark fixed demo

all: benchmark fixed demo

benchmark:
	$(CC) $(CFLAGS) benchmark.cpp  ../src/*.cpp ../src/*/*.cpp ../3rdParty/*/*.cpp -O3 -o benchmark -I ..
 
fixed:
	$(CC) $(CFLAGS) fixed_point.cpp  ../src/*.cpp ../src/*/*.cpp ../3rdParty/*/*.cpp -O3 -o fixed_point -I ..

demo:
	$(CC) $(CFLAGS) demo.cpp  ../src/*.cpp ../src/*/*.cpp ../3rdParty/*/*.cpp -O3 -o demo -I ..
