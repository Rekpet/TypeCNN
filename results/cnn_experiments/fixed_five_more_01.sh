echo "== Training on fixed point starting from base net for 5 more epochs (0.01) =="

echo "=================== Double ===================="
cd ../../
g++ -std=c++14 cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_cnn_five_more -I . -Wall -Wextra -DCNN_FTYPE="double" -DCNN_BTYPE="double" -DCNN_WTYPE="double"
cd results/cnn_experiments/
./../../TypeCNN_cnn_five_more --type-info -c net.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 5 -s 8786 --optimizer sgd --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-save -l 0.01

echo "=================== Float ===================="
cd ../../
g++ -std=c++14 cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_cnn_five_more -I . -Wall -Wextra -DCNN_FTYPE="float" -DCNN_BTYPE="float" -DCNN_WTYPE="float"
cd results/cnn_experiments/
./../../TypeCNN_cnn_five_more --type-info -c net.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 5 -s 8786 --optimizer sgd --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-save -l 0.01

echo "============= FixedPoint<16,16> =============="
cd ../../
g++ -std=c++14 cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_cnn_five_more -I . -Wall -Wextra -DCNN_FTYPE="FixedPoint64<16,16>" -DCNN_BTYPE="float" -DCNN_WTYPE="FixedPoint64<16,16>"
cd results/cnn_experiments/
./../../TypeCNN_cnn_five_more --type-info -c net.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 5 -s 8786 --optimizer sgd --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-save -l 0.01

echo "============= FixedPoint<8,8> =============="
cd ../../
g++ -std=c++14 cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_cnn_five_more -I . -Wall -Wextra -DCNN_FTYPE="FixedPoint<8,8>" -DCNN_BTYPE="float" -DCNN_WTYPE="FixedPoint<8,8>"
cd results/cnn_experiments/
./../../TypeCNN_cnn_five_more --type-info -c net.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 5 -s 8786 --optimizer sgd --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-save -l 0.01

echo "============= FixedPoint<4,4> =============="
cd ../../
g++ -std=c++14 cli/main.cpp src/*.cpp src/*/*.cpp 3rdParty/*/*.cpp -O3 -o TypeCNN_cnn_five_more -I . -Wall -Wextra -DCNN_FTYPE="FixedPoint<4,4>" -DCNN_BTYPE="float" -DCNN_WTYPE="FixedPoint<4,4>"
cd results/cnn_experiments/
./../../TypeCNN_cnn_five_more --type-info -c net.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 5 -s 8786 --optimizer sgd --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-save -l 0.01
