echo "================= MNIST run =============="
echo "Make sure the optimizer is: SGDM 0.005, 0.9, 0.0"
cd ../../
g++ -std=c++14 cli/main.cpp  src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_mnist -I . -Wall -Wextra -DCNN_BTYPE=float -DCNN_FTYPE=float -DCNN_WTYPE=float
cd results/99_25_mnist_lenet/
./../../TypeCNN_mnist -c lenet5.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 10 -s 1 --optimizer sgdm --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-load --shuffle --keep-best
