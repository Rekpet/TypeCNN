echo "================= CIFAR run =============="
echo "Make sure that optimizer is: SGDM 0.01, 0.9, 0.001"
cd ../../
g++ -std=c++14 cli/main.cpp  src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_cifar -I . -Wall -Wextra -DCNN_BTYPE=float -DCNN_FTYPE=float -DCNN_WTYPE=float
cd results/62_20_cifar10_lenet/
./../../TypeCNN_cifar -c net.xml -v ../../resources/cifar10/test_batch.bin -t ../../resources/cifar10/data_batch.bin -e 20 -s 2 --periodic-output 10000 --optimizer sgdm --loss-function MSE --do-not-load --periodic-validation --keep-best --shuffle -b 4