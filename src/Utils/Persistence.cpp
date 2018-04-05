/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Persistence module for CNN
 */

#include "src/Utils/Persistence.h"

#include "src/Utils/PersistenceMapper.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <regex>

using namespace PersistenceMapper;

/*
 * @brief Dumps CNN into given directory (cnn.xml + files with weights/filters)
 */
void Persistence::dumpNetwork(const ConvolutionalNeuralNetwork & cnn, const std::string & pathToXmlFile)
{
	std::smatch match;
	if (std::regex_search(pathToXmlFile.begin(), pathToXmlFile.end(), match, std::regex("(.*(/|\\\\))")))
	{
		directory = match[0];
	}

	tinyxml2::XMLDocument document;
	auto docRoot = document.NewElement("convolutional_neural_network");
	auto settingsRoot = document.NewElement("settings");
	auto architectureRoot = document.NewElement("architecture");

	dumpSettings(settingsRoot, document, cnn);
	dumpArchitecture(architectureRoot, document, cnn);

	docRoot->InsertEndChild(settingsRoot);
	docRoot->InsertEndChild(architectureRoot);
	document.InsertFirstChild(docRoot);

	auto returnCode = document.SaveFile(pathToXmlFile.c_str());
	if (returnCode != tinyxml2::XML_SUCCESS)
	{
		throw CannotCreateFilesOnDisk("Could not save XML file with architecture on disk.");
	}
}


/*
 * @brief Loads CNN from given xml file (expects weight/filter files in the same directory)
 */
ConvolutionalNeuralNetwork Persistence::loadNetwork(const std::string & pathToXmlFile, const bool lw)
{
	std::smatch match;
	if (std::regex_search(pathToXmlFile.begin(), pathToXmlFile.end(), match, std::regex("(.*(/|\\\\))")))
	{
		directory = match[0];
	}

	loadWeights = lw;
	settings = ParsedSettings();
	layerDumpIndex = 0;

	tinyxml2::XMLDocument document;

	auto returnCode = document.LoadFile(pathToXmlFile.c_str());
	if (returnCode != tinyxml2::XML_SUCCESS) {
		throw CouldNotParseXmlFile("Could not load or parse input XML file.");
	}

	auto docRoot = document.FirstChildElement();
	if (!docRoot || "convolutional_neural_network" != std::string(docRoot->Value())) {
		throw CouldNotParseXmlFile("XML content is not valid architecture. Expected \"convolutional_neural_network\" node.");
	}

	auto settingsRoot = docRoot->FirstChild();
	if (!settingsRoot || "settings" != std::string(settingsRoot->Value()))
	{
		throw CouldNotParseXmlFile("XML content is not valid architecture. Expected \"settings\" node.");
	}

	auto architectureRoot = settingsRoot->NextSibling();
	if (!architectureRoot || "architecture" != std::string(architectureRoot->Value()))
	{
		throw CouldNotParseXmlFile("XML content is not valid architecture. Expected \"architecture\" node.");
	}

	try
	{
		parseSettings(settingsRoot->ToElement());
	}
	catch (std::exception & e)
	{
		throw InvalidConvolutionalNeuralNetwork(e.what());
	}

	try
	{
		ConvolutionalNeuralNetwork cnn = parseArchitecture(architectureRoot->ToElement());
		return cnn;
	}
	catch (std::exception & e)
	{
		throw InvalidConvolutionalNeuralNetwork(e.what());
	}
}


/*
 * @brief Parses settings node in XML file
 */
void Persistence::parseSettings(tinyxml2::XMLElement * settingsRoot)
{
	auto currentNode = settingsRoot->FirstChild();
	unsigned parsedCnt = 0;
	while (currentNode)
	{
		auto name = std::string(currentNode->Value());
		auto currentElement = currentNode->ToElement();

		if (name == "task")
		{
			settings.taskType = getTaskType(currentElement->Attribute("type"));

			parsedCnt++;
		}
		else if (name == "input")
		{
			settings.input.width = std::stoi(currentElement->Attribute("width"));
			settings.input.height = std::stoi(currentElement->Attribute("height"));
			settings.input.depth = std::stoi(currentElement->Attribute("depth"));

			parsedCnt++;
		}
		else if (name == "output")
		{
			settings.output.width = std::stoi(currentElement->Attribute("width"));
			settings.output.height = std::stoi(currentElement->Attribute("height"));
			settings.output.depth = std::stoi(currentElement->Attribute("depth"));

			parsedCnt++;
		}
		else
		{
			throw InvalidConvolutionalNeuralNetwork("Unknows settings are present in XML file.");
		}

		currentNode = currentNode->NextSibling();
	}

	if (parsedCnt < 3)
	{
		throw InvalidConvolutionalNeuralNetwork("Not all required settings were found in XML file.");
	}
}


/*
 * @brief Parses architecture of CNN and creates CNN itself
 */
ConvolutionalNeuralNetwork Persistence::parseArchitecture(tinyxml2::XMLElement * architectureRoot)
{
	ConvolutionalNeuralNetwork cnn(settings.taskType);

	auto currentNode = architectureRoot->FirstChild();
	std::shared_ptr<ILayer<ForwardType, WeightType>> prevLayer = nullptr;

	while (currentNode)
	{
		auto name = std::string(currentNode->Value());
		auto currentElement = currentNode->ToElement();

		if (name == "layer")
		{
			std::string layerType = currentElement->Attribute("type");
			std::shared_ptr<ILayer<ForwardType, WeightType>> layer;
			if (layerType[0] != 'D') // disabled
			{
				if (layerType == "convolutional")
				{
					layer = parseConvolutionalLayer(currentElement, prevLayer);
				}
				else if (layerType == "pooling")
				{
					layer = parsePoolingLayer(currentElement, prevLayer);
				}
				else if (layerType == "fully_connected")
				{
					layer = parseFullyConnectedLayer(currentElement, prevLayer);
				}
				else if (layerType == "dropout")
				{
					layer = parseDropoutLayer(currentElement, prevLayer);
				}
				else if (layerType == "activation")
				{
					layer = parseActivationLayer(currentElement, prevLayer);
				}
				else
				{
					throw InvalidConvolutionalNeuralNetwork("Unexpected layer found in architecture.");
				}
				cnn.addLayer(layer);
				prevLayer = layer;
			}
		}
		else
		{
			throw InvalidConvolutionalNeuralNetwork("Unexpected node in architecture.");
		}

		currentNode = currentNode->NextSibling();
	}

	if (prevLayer->getOutputSize() != settings.output)
	{
		throw InvalidConvolutionalNeuralNetwork("Last layer size is not the same as declared output size.");
	}

	return cnn;
}


/*
 * @brief Parses and creates Convolutional layer
 */
std::shared_ptr<ILayer<ForwardType, WeightType>> Persistence::parseConvolutionalLayer(tinyxml2::XMLElement * root, std::shared_ptr<ILayer<ForwardType, WeightType>> prevLayer)
{
	std::string pathToFilters;
	unsigned filterNum = 0;
	unsigned filterExtent = 0;
	unsigned stride = 0;
	unsigned zeroPadding = 0;
	bool useBias = false;

	auto currentNode = root->FirstChild();
	while (currentNode)
	{
		auto name = std::string(currentNode->Value());
		auto currentElement = currentNode->ToElement();

		if (name == "bias")
		{
			std::string useBiasStr = currentElement->Attribute("use");
			useBias = useBiasStr == "true";
		}
		else if (name == "stride")
		{
			std::string val = currentElement->Attribute("value");
			stride = static_cast<unsigned>(std::stoi(val));
		}
		else if (name == "zero_padding")
		{
			std::string val = currentElement->Attribute("value");
			zeroPadding = static_cast<unsigned>(std::stoi(val));
		}
		else if (name == "filters")
		{
			std::string val = currentElement->Attribute("extent");
			filterExtent = static_cast<unsigned>(std::stoi(val));

			val = currentElement->Attribute("number");
			filterNum = static_cast<unsigned>(std::stoi(val));

			if (currentElement->Attribute("path"))
			{
				pathToFilters = directory + currentElement->Attribute("path");
			}
		}
		else
		{
			throw InvalidConvolutionalNeuralNetwork("Unexpected node in Convolutional layer definition.");
		}

		currentNode = currentNode->NextSibling();
	}

	if (stride == 0 || filterExtent == 0 || filterNum == 0)
	{
		throw InvalidConvolutionalNeuralNetwork("Mandatory settings are missing in Convolutional layer definition.");
	}

	Dimensions inputDimension;
	if (prevLayer)
	{
		inputDimension = prevLayer->getOutputSize();
	}
	else
	{
		inputDimension = settings.input;
	}

	auto layer = std::make_shared<ConvolutionalLayer<ForwardType, WeightType>>(inputDimension, stride, filterNum, filterExtent, zeroPadding, useBias);

	if (!pathToFilters.empty() && loadWeights)
	{
		auto parsedFilters = parseFilters(pathToFilters, filterNum, filterExtent, inputDimension.depth);
		layer->loadFilters(parsedFilters.first, parsedFilters.second);
	}

	return layer;
}


/*
 * @brief Parses and creates Pooling layer
 */
std::shared_ptr<ILayer<ForwardType, WeightType>> Persistence::parsePoolingLayer(tinyxml2::XMLElement * root, std::shared_ptr<ILayer<ForwardType, WeightType>> prevLayer)
{
	PoolingOperation operationType = PoolingOperation::Max;
	unsigned stride = 0;
	unsigned extentSize = 0;

	auto currentNode = root->FirstChild();
	while (currentNode)
	{
		auto name = std::string(currentNode->Value());
		auto currentElement = currentNode->ToElement();

		if (name == "operation")
		{
			operationType = getPoolingOperationType(currentElement->Attribute("type"));
		}
		else if (name == "stride")
		{
			std::string opType = currentElement->Attribute("value");
			stride = static_cast<unsigned>(std::stoi(opType));
		}
		else if (name == "extent")
		{
			std::string opType = currentElement->Attribute("value");
			extentSize = static_cast<unsigned>(std::stoi(opType));
		}
		else
		{
			throw InvalidConvolutionalNeuralNetwork("Unexpected node in Pooling layer definition.");
		}

		currentNode = currentNode->NextSibling();
	}

	if (stride == 0 || extentSize == 0)
	{
		throw InvalidConvolutionalNeuralNetwork("Stride and extent size cannot be zero or lower.");
	}

	Dimensions inputDimension;
	if (prevLayer)
	{
		inputDimension = prevLayer->getOutputSize();
	}
	else
	{
		inputDimension = settings.input;
	}

	switch (operationType)
	{
		case PoolingOperation::Average:
			return std::make_shared<AvgPooling>(inputDimension, extentSize, stride);
		case PoolingOperation::Max: default:
			return std::make_shared<MaxPooling>(inputDimension, extentSize, stride);
	}
}


/*
 * @brief Parses and creates Dropout layer
 */
std::shared_ptr<ILayer<ForwardType, WeightType>> Persistence::parseDropoutLayer(tinyxml2::XMLElement * root, std::shared_ptr<ILayer<ForwardType, WeightType>> prevLayer)
{
	float probability = -1.0f;

	auto currentNode = root->FirstChild();
	while (currentNode)
	{
		auto name = std::string(currentNode->Value());
		auto currentElement = currentNode->ToElement();

		if (name == "probability")
		{
			probability = std::stof(currentElement->Attribute("value"));
		}
		else
		{
			throw InvalidConvolutionalNeuralNetwork("Unexpected node in Dropout layer definition.");
		}

		currentNode = currentNode->NextSibling();
	}

	if (probability < 0)
	{
		throw InvalidConvolutionalNeuralNetwork("Dropout probability not set in Dropout layer.");
	}

	Dimensions inputDimension;
	if (prevLayer)
	{
		inputDimension = prevLayer->getOutputSize();
	}
	else
	{
		inputDimension = settings.input;
	}

	auto layer = std::make_shared<DropoutLayer<ForwardType, WeightType>>(inputDimension, probability);

	return layer;
}


/*
 * @brief Parses and creates Fully Connected layer
 */
std::shared_ptr<ILayer<ForwardType, WeightType>> Persistence::parseFullyConnectedLayer(tinyxml2::XMLElement * root, std::shared_ptr<ILayer<ForwardType, WeightType>> prevLayer)
{
	Dimensions outputSize{0,1,1};
	std::string pathToWeights;
	bool useBias = true;

	auto currentNode = root->FirstChild();
	while (currentNode)
	{
		auto name = std::string(currentNode->Value());
		auto currentElement = currentNode->ToElement();

		if (name == "bias")
		{
			std::string useBiasStr = currentElement->Attribute("use");
			useBias = useBiasStr == "true";
		}
		else if (name == "weights")
		{
			if (currentElement->Attribute("path"))
			{
				pathToWeights = directory + currentElement->Attribute("path");
			}
		}
		else if (name == "output_layer")
		{
			outputSize.width = static_cast<unsigned>(std::stoi(currentElement->Attribute("size")));
		}
		else
		{
			throw InvalidConvolutionalNeuralNetwork("Unexpected node in Fully connected layer definition.");
		}

		currentNode = currentNode->NextSibling();
	}

	if (outputSize.width == 0)
	{
		throw InvalidConvolutionalNeuralNetwork("Outpit size of dense layer not set or set to zero.");
	}

	Dimensions inputDimension;
	if (prevLayer)
	{
		inputDimension = prevLayer->getOutputSize();
	} 
	else
	{
		inputDimension = settings.input;
	}

	auto layer = std::make_shared<FullyConnectedLayer<ForwardType, WeightType>>(inputDimension, outputSize, useBias);
	
	if (!pathToWeights.empty() && loadWeights)
	{
		layer->setNeuronWeights(parseWeights(pathToWeights, inputDimension.width * inputDimension.height * inputDimension.depth, outputSize.width * outputSize.height * outputSize.depth));
	}

	return layer;
}


/*
 * @brief Parses and creates Activation layer
 */
std::shared_ptr<ILayer<ForwardType, WeightType>> Persistence::parseActivationLayer(tinyxml2::XMLElement * root, std::shared_ptr<ILayer<ForwardType, WeightType>> prevLayer)
{
	ActivationFunction activationFunction = ActivationFunction::Sigmoid;

	auto currentNode = root->FirstChild();
	while (currentNode)
	{
		auto name = std::string(currentNode->Value());
		auto currentElement = currentNode->ToElement();

		if (name == "activation")
		{
			activationFunction = getActivationFunctionType(currentElement->Attribute("type"));
		}
		else
		{
			throw InvalidConvolutionalNeuralNetwork("Unexpected node in Activation layer definition.");
		}

		currentNode = currentNode->NextSibling();
	}

	Dimensions inputDimension;
	if (prevLayer)
	{
		inputDimension = prevLayer->getOutputSize();
	}
	else
	{
		inputDimension = settings.input;
	}

	return PersistenceMapper::getActivationLayer(activationFunction, inputDimension);
}


/*
 * @brief Dumps settings to output XML file
 */
void Persistence::dumpSettings(tinyxml2::XMLNode * settingsRoot, tinyxml2::XMLDocument & document, const ConvolutionalNeuralNetwork & cnn)
{
	auto inputRoot = document.NewElement("input");
	auto outputRoot = document.NewElement("output");
	auto taskRoot = document.NewElement("task");

	outputRoot->SetAttribute("width", cnn.getOutputSize().width);
	outputRoot->SetAttribute("height", cnn.getOutputSize().height);
	outputRoot->SetAttribute("depth", cnn.getOutputSize().depth);

	taskRoot->SetAttribute("type", getTaskTypeString(cnn.getTaskType()).c_str());

	inputRoot->SetAttribute("width", cnn.getInputSize().width);
	inputRoot->SetAttribute("height", cnn.getInputSize().height);
	inputRoot->SetAttribute("depth", cnn.getInputSize().depth);

	settingsRoot->InsertEndChild(taskRoot);
	settingsRoot->InsertEndChild(inputRoot);
	settingsRoot->InsertEndChild(outputRoot);
}


/*
 * @brief Dumps architecture and weights/filters of given CNN
 */
void Persistence::dumpArchitecture(tinyxml2::XMLNode * architectureRoot, tinyxml2::XMLDocument & document, const ConvolutionalNeuralNetwork & cnn)
{
	for (const auto layer : cnn)
	{
		auto layerRoot = document.NewElement("layer");
		layerDumpIndex++;

		if (auto convPtr = dynamic_cast<ConvolutionalLayer<ForwardType, WeightType> *>(layer.get()))
		{
			layerRoot->SetAttribute("type", "convolutional");
			dumpConvolutionalLayer(layerRoot, document, convPtr);
		}
		else if (auto poolPtr = dynamic_cast<PoolingLayer<ForwardType, WeightType> *>(layer.get()))
		{
			layerRoot->SetAttribute("type", "pooling");
			dumpPoolingLayer(layerRoot, document, poolPtr);
		}
		else if (auto fullPtr = dynamic_cast<FullyConnectedLayer<ForwardType, WeightType> *>(layer.get()))
		{
			layerRoot->SetAttribute("type", "fully_connected");
			dumpFullyConnectedLayer(layerRoot, document, fullPtr);
		}
		else if (auto dropPtr = dynamic_cast<DropoutLayer<ForwardType, WeightType> *>(layer.get()))
		{
			layerRoot->SetAttribute("type", "dropout");
			dumpDropoutLayer(layerRoot, document, dropPtr);
		}
		else if (auto actPtr = dynamic_cast<ActivationLayer<ForwardType, WeightType> *>(layer.get()))
		{
			layerRoot->SetAttribute("type", "activation");
			dumpActivationLayer(layerRoot, document, actPtr);
		}
		else
		{
			throw InvalidConvolutionalNeuralNetwork("Could not dump one of the layers as it is not supported by Persistence module.");
		}

		architectureRoot->InsertEndChild(layerRoot);
	}
}


/*
 * @brief Dumps convolutional layer properties
 */
void Persistence::dumpConvolutionalLayer(tinyxml2::XMLNode * layerRoot, tinyxml2::XMLDocument & document, ConvolutionalLayer<ForwardType, WeightType> * layer)
{
	auto strideRoot = document.NewElement("stride");
	auto zeroPaddingRoot = document.NewElement("zero_padding");
	auto filtersRoot = document.NewElement("filters");
	auto biasRoot = document.NewElement("bias");

	if (layer->usesBias())
	{
		biasRoot->SetAttribute("use", "true");
	}
	else
	{
		biasRoot->SetAttribute("use", "false");
	}

	auto filtersFileName = std::to_string(layerDumpIndex) + "_conv_layer.txt";
	dumpFilters<WeightType>(directory + filtersFileName, layer->getFilters(), layer->getBiases());
	filtersRoot->SetAttribute("path", filtersFileName.c_str());
	filtersRoot->SetAttribute("number", layer->getFilterNum());
	filtersRoot->SetAttribute("extent", layer->getExtent());

	strideRoot->SetAttribute("value", layer->getStride());
	zeroPaddingRoot->SetAttribute("value", layer->getZeroPadding());

	layerRoot->InsertEndChild(strideRoot);
	layerRoot->InsertEndChild(zeroPaddingRoot);
	layerRoot->InsertEndChild(filtersRoot);
	layerRoot->InsertEndChild(biasRoot);
}


/*
 * @brief Dumps pooling layer properties
 */
void Persistence::dumpPoolingLayer(tinyxml2::XMLNode * layerRoot, tinyxml2::XMLDocument & document, PoolingLayer<ForwardType, WeightType> * layer)
{
	auto operationRoot = document.NewElement("operation");
	auto strideRoot = document.NewElement("stride");
	auto extentRoot = document.NewElement("extent");

	operationRoot->SetAttribute("type", getPoolingOperationString(layer->getPoolingOperationType()).c_str());

	strideRoot->SetAttribute("value", layer->getStride());
	extentRoot->SetAttribute("value", layer->getExtentSize());

	layerRoot->InsertEndChild(operationRoot);
	layerRoot->InsertEndChild(strideRoot);
	layerRoot->InsertEndChild(extentRoot);
}


/*
 * @brief Dumps dropout layer properties
 */
void Persistence::dumpDropoutLayer(tinyxml2::XMLNode * layerRoot, tinyxml2::XMLDocument & document, DropoutLayer<ForwardType, WeightType> * layer)
{
	auto probabilityRoot = document.NewElement("probability");
	probabilityRoot->SetAttribute("value", layer->getDropoutProbability());
	layerRoot->InsertEndChild(probabilityRoot);
}


/*
 * @brief Dumps fully connected layer properties
 */
void Persistence::dumpFullyConnectedLayer(tinyxml2::XMLNode * layerRoot, tinyxml2::XMLDocument & document, FullyConnectedLayer<ForwardType, WeightType> * layer)
{
	auto outputRoot = document.NewElement("output_layer");
	auto weightsRoot = document.NewElement("weights");
	auto biasRoot = document.NewElement("bias");

	auto weightsFileName = std::to_string(layerDumpIndex) + "_fc_layer.txt";
	dumpWeights<WeightType>(directory + weightsFileName, layer->getNeuronWeights());
	weightsRoot->SetAttribute("path", weightsFileName.c_str());

	outputRoot->SetAttribute("size", layer->getOutputSize().width * layer->getOutputSize().height * layer->getOutputSize().depth);

	if (layer->usesBias())
	{
		biasRoot->SetAttribute("use", "true");
	}
	else
	{
		biasRoot->SetAttribute("use", "false");
	}

	layerRoot->InsertEndChild(outputRoot);
	layerRoot->InsertEndChild(weightsRoot);
	layerRoot->InsertEndChild(biasRoot);
}


/*
 * @brief Dumps activation layer properties
 */
void Persistence::dumpActivationLayer(tinyxml2::XMLNode * layerRoot, tinyxml2::XMLDocument & document, ActivationLayer<ForwardType, WeightType> * layer)
{
	auto activationRoot = document.NewElement("activation");

	activationRoot->SetAttribute("type", getActivationFunctionString(layer->getActivationFunctionType()).c_str());

	layerRoot->InsertEndChild(activationRoot);
}


/*
 * @brief Dumps weights of Fully Connected Layer to file
 */
template <class OutType>
void Persistence::dumpWeights(const std::string & pathToWeights, const Image<BackwardType> & weights)
{
	std::ofstream output(pathToWeights);

	auto flattenedSize = weights.getFlattenedSize();

	if (output.is_open())
	{
		for (auto i = 0u; i < flattenedSize; i++)
		{
			output << std::setprecision(30) << static_cast<OutType>(weights(i)) << "\n";
		}
	}
	else
	{
		throw CannotCreateFilesOnDisk("Could not create file with weights for Fully connected layer.");
	}
}


/*
 * @brief Parses weights for Fully Connected layer from file
 */
Image<BackwardType> Persistence::parseWeights(const std::string & pathToWeights, const unsigned inputNeurons, const unsigned outputNeurons)
{
	std::ifstream input(pathToWeights);
	Image<BackwardType> weights{ Dimensions{ inputNeurons + 1, outputNeurons, 1 } };

	auto cnt = 0u;
	if (input.is_open())
	{
		std::string line;
		while (std::getline(input, line))
		{
			if (!line.empty())
			{
				weights(cnt++) = static_cast<BackwardType>(std::stof(line));
			}
		}
	}
	else
	{
		throw InvalidWeights("Could not load weights for Fully connected layer.");
	}

	return weights;
}


/*
 * @brief Dumps filters of Convolutional Neural Network
 */
template <class OutType>
void Persistence::dumpFilters(const std::string & pathToFilters, const std::vector<Image<BackwardType>> & filters, const std::vector<BackwardType> & biases)
{
	std::ofstream output(pathToFilters);

	if (output.is_open())
	{
		for (const auto & filter : filters)
		{
			auto dim = filter.getDimensions();

			for (auto depth = 0u; depth < dim.depth; depth++)
			{
				for (auto height = 0u; height < dim.height; height++)
				{
					for (auto width = 0u; width < dim.width; width++)
					{
						output << std::setprecision(30) << static_cast<OutType>(filter(width, height, depth)) << " ";
					}
					output << "\n"; // not using std::endl to be consistent on all OS
				}
				output << "\n";
			}
			output << "\n";
		}

		output << "\n";

		for (auto bias : biases)
		{
			output << std::setprecision(30) << static_cast<OutType>(bias) << "\n";;
		}
	}
	else
	{
		throw CannotCreateFilesOnDisk("Could not save filters for Convolutional layer.");
	}
}


/*
 * @brief Parses fiters for Convolutional layer
 */
std::pair<std::vector<Image<BackwardType>>, std::vector<BackwardType>> Persistence::parseFilters(
	const std::string & pathToFilters, const unsigned & filterNum, const unsigned & extent, const unsigned & inputDepth)
{
	std::ifstream input(pathToFilters);
	std::vector<Image<BackwardType>> filters;
	std::vector<BackwardType> biases;

	if (input.is_open())
	{
		std::string line;

		for (auto filter = 0u; filter < filterNum; filter++)
		{
			Image<BackwardType> filterPtr(Dimensions{ extent, extent, inputDepth });
			for (auto depth = 0u; depth < inputDepth; depth++)
			{
				for (auto height = 0u; height < extent; height++)
				{
					std::getline(input, line);
					auto splittedLine = splitLineByDelimiter(line, ' ');

					if (splittedLine.size() != extent)
					{
						throw InvalidConvolutionalNeuralNetwork("Could not load filters due inconsistent size.");
					}

					for (auto width = 0u; width < extent; width++)
					{
						filterPtr(width, height, depth) = static_cast<BackwardType>(std::stof(splittedLine[width]));
					}
				}
				std::getline(input, line); // skip empty line
			}
			filters.push_back(filterPtr);
			std::getline(input, line); // skip empty line
		}

		std::getline(input, line); // skip empty line

		for (auto filter = 0u; filter < filterNum; filter++)
		{
			if (std::getline(input, line))
			{
				biases.push_back(static_cast<BackwardType>(std::stof(line)));
			}
			else
			{
				throw InvalidConvolutionalNeuralNetwork("Could not load filters due inconsistent size.");
			}
		}
	}
	else
	{
		throw InvalidFilters("Could not open file to load filters for Convolutional layer.");
	}

	return { filters, biases };
}


/*
 * @brief Splits line by delimiter character
 */
std::vector<std::string> Persistence::splitLineByDelimiter(const std::string & line, const char & delimiter)
{
	std::istringstream ss(line);
	std::string s;
	std::vector<std::string> out;

	while (getline(ss, s, delimiter))
	{
		if (s != "\r") // inconsistency Windows <-> Linux
		{
			out.push_back(s);
		}
	}

	return out;
}