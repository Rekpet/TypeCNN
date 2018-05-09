# TypeCNN

TypeCNN is a convolutional neural network library that provides reasonable amount of functionality and reasonable speed on CPU. It as also data type independent and thus allows you to experiment with different types. Most notably it contains an implementation of fixed point type that has definable bit length. Experiments were performed with float and fixed point representations and the results can be found in "results/" folder.

Currently the only documentation is in Czech [1]. Refer to "src/CommandLineInterface.cpp" or "examples/" for examples of usage (should be quite simple).

## Usage of command line interface

```
TypeCNN - data type independent Convolutional Neural Network library.
Usage:
  TypeCNN [OPTION...]

 Common options:
  -h, --help       Shows this help message.
  -c, --cnn FILE   Input XML file with CNN description.
  -g, --grayscale  Specifies that we are working with grayscale PNG images.
      --type-info  Shows info about types used.

 Inference options:
  -i, --input FILE  Input PNG image for inference.

 Validation options:
  -v, --validate FILE(s)      Validation data files separated with space.
      --validate-offset UINT  Offset into validation data (how much to skip).
      --validate-num UINT     How much validation data to use, 0 == all.

 Training options:
  -t, --train FILE(s)         Training data files separated with space.
      --train-offset UINT     Offset into training data (how much to skip).
      --train-num UINT        How much training data to use, 0 == all.
  -s, --seed UINT             Seed for random generator.
  -e, --epochs UINT           Number of epochs for training.
  -l, --learning-rate DOUBLE  Learning coefficient.
  -d, --weight-decay DOUBLE   Weight decay coefficient.
  -b, --batch-size UINT       Batch size.
      --do-not-load           Do not load weights.
      --do-not-save           Do not save weights after training.
      --optimizer TYPE        Optimizer type (sgd|sgdm|sgdn|adam|adagrad).
      --loss-function TYPE    Loss function to be used (MSE|CE|CEbin).
      --periodic-validation   Runs validation before and after each epoch.
      --periodic-output UINT  Outputs average error of each X samples.
      --shuffle               Shuffle training data before each epoch begins.
      --keep-best             Saves trained network with highest validation
                              accuracy during training.

```

[1] REK, Petr. Knihovna pro návrh konvolučních neuronových sítí. Brno, 2018. Diplomová
práce. Vysoké učení technické v Brně, Fakulta informačních technologií. Vedoucí práce prof.
Ing. Lukáš Sekanina, Ph.D.