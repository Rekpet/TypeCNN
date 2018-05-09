echo "======================== Base net training ========================"
cd ../../
g++ -std=c++14 cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_nn_five_more -I . -Wall -Wextra -DCNN_FTYPE="float" -DCNN_BTYPE="float" -DCNN_WTYPE="float"
cd results/nn_experiments/
./../../TypeCNN_nn_five_more --type-info -c net.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 10 -s 456852 --optimizer sgdm --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-load --shuffle