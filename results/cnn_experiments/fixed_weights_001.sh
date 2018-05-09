echo "== 8,8 for inference, given type for weights (0.001) =="

echo "============= FixedPoint<8,8> =============="
cd ../../
g++ -std=c++14 cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_cnn_fixed_weights -I . -Wall -Wextra -DCNN_FTYPE="FixedPoint<8,8>" -DCNN_BTYPE="float" -DCNN_WTYPE="FixedPoint<8,8>"
cd results/cnn_experiments/
./../../TypeCNN_cnn_fixed_weights --type-info -c net.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 5 -s 8786 --optimizer sgd --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-save -l 0.001

echo "============= FixedPoint<4,4> =============="
cd ../../
g++ -std=c++14 cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_cnn_fixed_weights -I . -Wall -Wextra -DCNN_FTYPE="FixedPoint<8,8>" -DCNN_BTYPE="float" -DCNN_WTYPE="FixedPoint<4,4>"
cd results/cnn_experiments/
./../../TypeCNN_cnn_fixed_weights --type-info -c net.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 5 -s 8786 --optimizer sgd --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-save -l 0.001

echo "============= FixedPoint<2,2> =============="
cd ../../
g++ -std=c++14 cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_cnn_fixed_weights -I . -Wall -Wextra -DCNN_FTYPE="FixedPoint<8,8>" -DCNN_BTYPE="float" -DCNN_WTYPE="FixedPoint<2,2>"
cd results/cnn_experiments/
./../../TypeCNN_cnn_fixed_weights --type-info -c net.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 5 -s 8786 --optimizer sgd --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-save -l 0.001

echo "============= FixedPoint<1,3> =============="
cd ../../
g++ -std=c++14 cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_cnn_fixed_weights -I . -Wall -Wextra -DCNN_FTYPE="FixedPoint<8,8>" -DCNN_BTYPE="float" -DCNN_WTYPE="FixedPoint<1,3>"
cd results/cnn_experiments/
./../../TypeCNN_cnn_fixed_weights --type-info -c net.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 5 -s 8786 --optimizer sgd --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-save -l 0.001

echo "============= FixedPoint<1,1> =============="
cd ../../
g++ -std=c++14 cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_cnn_fixed_weights -I . -Wall -Wextra -DCNN_FTYPE="FixedPoint<8,8>" -DCNN_BTYPE="float" -DCNN_WTYPE="FixedPoint<1,1>"
cd results/cnn_experiments/
./../../TypeCNN_cnn_fixed_weights --type-info -c net.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 5 -s 8786 --optimizer sgd --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-save -l 0.001
