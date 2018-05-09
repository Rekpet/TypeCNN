echo "================= Base net training =============="
echo "Optimizer: SGDM (learning rate 0.005, weight decay 0.0, momentum 0.9)"
echo "Using Mean Squared Error"
echo "Training on float"
./../../Thesis -c lenet5.xml -v ../../resources/mnist/test-images.idx3-ubyte -t ../../resources/mnist/train-images.idx3-ubyte -e 13 -s 1 --optimizer sgdm --loss-function MSE --periodic-validation --periodic-output 5000 --do-not-load --shuffle --keep-best -l 0.005
