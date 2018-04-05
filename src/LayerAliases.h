/*
* @author Petr Rek
* @project CNN Library
* @brief Defines layer aliases for simple usage
*/

#ifdef _MSC_VER
#pragma once
#endif

#ifndef LAYER_ALIASES_H
#define LAYER_ALIASES_H

#include "src/CompileSettings.h"

#include "src/Layers/ConvolutionalLayer.h"

#include "src/Layers/DropoutLayer.h"

#include "src/Layers/FullyConnectedLayer.h"

#include "src/Layers/MaxPoolingLayer.h"
#include "src/Layers/AvgPoolingLayer.h"

#include "src/Layers/ActivationLayer.h"
#include "src/Layers/LeakyReluActivationLayer.h"
#include "src/Layers/ReluActivationLayer.h"
#include "src/Layers/SigmoidActivationLayer.h"
#include "src/Layers/SoftmaxActivationLayer.h"
#include "src/Layers/TanhActivationLayer.h"

#include "src/Optimizers/Adagrad.h"
#include "src/Optimizers/Adam.h"
#include "src/Optimizers/Sgd.h"
#include "src/Optimizers/SgdWithMomentum.h"
#include "src/Optimizers/SgdWithNestorovMomentum.h"

/*
 * @brief Convolutional layer, CNN core
 */
using Convolution = ConvolutionalLayer<ForwardType, WeightType>;

/*
 * @brief Dropout layer, used during training to prevent overfitting
 */
using Dropout = DropoutLayer<ForwardType, WeightType>;

/*
 * @brief Fully connected layer -> neural network without hidden layers
 */
using FullyConnected = FullyConnectedLayer<ForwardType, WeightType>;

/*
 * @brief Pooling layers that reduce matrix size
 */
using MaxPooling = MaxPoolingLayer<ForwardType, WeightType>;
using AvgPooling = AvgPoolingLayer<ForwardType, WeightType>;

/*
 * @brief Activation layers (after FC or CONV usually)
 */
using LeakyReLU = LeakyReluActivationLayer<ForwardType, WeightType>;
using ReLU = ReluActivationLayer<ForwardType, WeightType>;
using Sigmoid = SigmoidActivationLayer<ForwardType, WeightType>;
using SoftMax = SoftmaxActivationLayer<ForwardType, WeightType>;
using Tanh = TanhActivationLayer<ForwardType, WeightType>;

#endif 