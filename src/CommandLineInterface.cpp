/*
 * @author Petr Rek
 * @project Convolutional Neural Network library
 * @brief CNN console interface
 */

#include "src/CommandLineInterface.h"

#include "src/Utils/ImageUtils.h"
#include "src/Parsers/IdxParser.h"
#include "src/Parsers/PngParser.h"
#include "src/Utils/Persistence.h"
#include "src/Parsers/BinaryParser.h"
#include "src/ConvolutionalNeuralNetwork.h"
#include "src/Utils/PersistenceMapper.h"

#include <regex>
#include <sstream>
#include <algorithm>

/*
 * @brief Command line interfasce sets up argument parser
 */
CommandLineInterface::CommandLineInterface()
	: options("CNN-library", "Command line interface for CNN-library.")
{
	// Set up argument parser
	options.add_options("Common")
		("h,help", "Shows this help message.")
		("c,cnn", "FILE", cxxopts::value<std::string>(), "Input XML file with CNN description.")
		("g,grayscale", "Specifies that we are working with grayscale PNG images.");
	options.add_options("Inference")
		("i,input", "FILE", cxxopts::value<std::string>(), "Input PNG image for inference.");
	options.add_options("Validation")
		("v,validate", "FILE(s)", cxxopts::value<std::vector<std::string>>(), "Validation data files separated with space.")
		("validate-offset", "UINT", cxxopts::value<unsigned>(), "Offset into validation data (how much to skip).")
		("validate-num", "UINT", cxxopts::value<unsigned>(), "How much validation data to use, 0 == all.");
	options.add_options("Training")
		("t,train", "FILE(s)", cxxopts::value<std::vector<std::string>>(), "Training data files separated with space.")
		("train-offset", "UINT", cxxopts::value<unsigned>(), "Offset into training data (how much to skip).")
		("train-num", "UINT", cxxopts::value<unsigned>(), "How much training data to use, 0 == all.")
		("s,seed", "UINT", cxxopts::value<unsigned>(), "Seed for random generator.")
		("e,epochs", "UINT", cxxopts::value<unsigned>(), "Number of epochs for training.")
		("l,learning-rate", "DOUBLE", cxxopts::value<float>(), "Learning coefficient - recommended range is (0, 1).")
		("b,batch-size", "UINT", cxxopts::value<unsigned>(), "Batch size (recommended value is 1).")
		("do-not-load", "Do not load weights.")
		("do-not-save", "Do not save weights after training.")
		("optimizer", "TYPE", cxxopts::value<std::string>(), "Optimizer to be used (sgd|sgdm|sgdn|adam|adagrad).")
		("loss-function", "TYPE", cxxopts::value<std::string>(), "Loss function to be used (MSE|CE).")
		("periodic-validation", "Runs validation before and after each epoch.")
		("periodic-output", "UINT", cxxopts::value<unsigned>(), "Outputs average error of each X samples.")
		("shuffle", "Shuffle training data before each epoch begins.")
		("keep-best", "Saves trained network with highest validation accuracy during training.");	
}


/*
 * @brief Runs CLI with given arguments
 */
int CommandLineInterface::runWithGivenArguments(int argc, char ** argv)
{
	// Save argc as it is modified by CxxOpts
	auto argcBackup = static_cast<unsigned>(argc);

	// If no args, print error and help
	if (argc == 1)
	{
		errorWhenParsingArguments("No parameters given.");
		return EXIT_FAILURE;
	}

	auto inference = false;
	auto inputInferencePath = std::string();

	auto validation = false;
	auto validationFiles = std::vector<std::string>();
	auto validationOffset = 0u;
	auto validationNum = 0u;

	auto training = false;
	auto randomSeed = static_cast<unsigned>(time(nullptr));
	auto trainingFiles = std::vector<std::string>();
	auto trainingOffset = 0u;
	auto trainingNum = 0u;
	auto trainingSettings = TrainingSettings();
	auto loadWeights = true;

	auto optimizer = PersistenceMapper::getOptimizerInstance(OptimizerType::Sgd);
	auto lossFunction = LossFunctionType::MeanSquaredError;

	// Parse arguments
	try
	{
		auto args = options.parse(argc, argv);

		if (args.count("help"))
		{
			std::cout << options.help({ "Common", "Inference", "Validation", "Training" }) << std::endl;
			return EXIT_SUCCESS;
		}

		if (args.count("grayscale"))
		{
			grayscale = true;
		}

		if (args.count("cnn"))
		{
			cnnPath = args["cnn"].as<std::string>();
		}
		else
		{
			errorWhenParsingArguments("XML representation of CNN required.");
			return EXIT_FAILURE;
		}

		if (args.count("input"))
		{
			inference = true;
			inputInferencePath = args["input"].as<std::string>();

			if (argcBackup > 6)
			{
				errorWhenParsingArguments("Invalid combination of parameters for Inference mode.");
				return EXIT_FAILURE;
			}
		}

		if (args.count("train"))
		{
			training = true;
			trainingFiles = args["train"].as<std::vector<std::string>>();

			if (args.count("optimizer"))
			{
				auto optimizerStr = args["optimizer"].as<std::string>();
				optimizer = PersistenceMapper::getOptimizerInstance(PersistenceMapper::getOptimizerType(optimizerStr));
			}

			if (args.count("shuffle"))
				trainingSettings.shuffle = true;

			if (args.count("learning-rate"))
				optimizer->learningRate = args["learning-rate"].as<float>();

			if (args.count("do-not-load"))
				loadWeights = false;

			if (args.count("do-not-save"))
				saveWeights = false;

			if (args.count("seed"))
				randomSeed = args["seed"].as<unsigned>();

			if (args.count("train-offset"))
				trainingOffset = args["train-offset"].as<unsigned>();

			if (args.count("train-num"))
				trainingNum = args["train-num"].as<unsigned>();

			if (args.count("epochs"))
				trainingSettings.epochs = args["epochs"].as<unsigned>();

			if (args.count("batch-size"))
				trainingSettings.batchSize = args["batch-size"].as<unsigned>();

			if (args.count("periodic-output"))
				trainingSettings.errorOutputRate = args["periodic-output"].as<unsigned>();

			if (args.count("periodic-validation"))
				trainingSettings.periodicValidation = true;

			if (args.count("keep-best"))
				keepBest = true;

			if (args.count("loss-function"))
			{
				auto lossFunctionStr = args["loss-function"].as<std::string>();
				lossFunction = PersistenceMapper::getLossFunctionType(lossFunctionStr);
			}

			if (!args.count("validate") && (args.count("validation-offset") || args.count("validation-num")))
			{
				errorWhenParsingArguments("Cannot set validation num/offset without setting validation files.");
				return EXIT_FAILURE;
			}
			else if (keepBest && !saveWeights)
			{
				errorWhenParsingArguments("Cannot keep best if saving is not enabled.");
				return EXIT_FAILURE;
			}
			else if (keepBest && !trainingSettings.periodicValidation)
			{
				errorWhenParsingArguments("Cannot keep best if periodic validation is not enabled.");
				return EXIT_FAILURE;
			}
		}

		if (args.count("validate"))
		{
			validation = true;
			validationFiles = args["validate"].as<std::vector<std::string>>();

			if (args.count("validate-offset"))
				validationOffset = args["validate-offset"].as<unsigned>();

			if (args.count("validate-num"))
				validationNum = args["validate-num"].as<unsigned>();

			if (!training && (2 * args.count("validate-offset") + 2 * args.count("validate-num") + validationFiles.size() + 4) != argcBackup)
			{
				errorWhenParsingArguments("Invalid combination of parameters for Validation mode.");
				return EXIT_FAILURE;
			}
		}

		if (!inference && !training && !validation) {
			errorWhenParsingArguments("No mode chosen. Choose either inference, training and/or validation.");
			return EXIT_FAILURE;
		}
		else if (inference && (training || validation))
		{
			errorWhenParsingArguments("Cannot run input mode along validation/training.");
			return EXIT_FAILURE;
		}
	}
	catch (const std::exception & e)
	{
		errorWhenParsingArguments(e.what());
		return EXIT_FAILURE;
	}

	// Initialize random seed
	srand(randomSeed);

	// Load CNN
	auto persistence = Persistence();
	try
	{
		cnn = persistence.loadNetwork(cnnPath, loadWeights);
		cnn.enableOutput();
	}
	catch (const PersistenceException & e)
	{
		std::cerr << "Could not load network from given file.\n  Reason: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// Run selected modes
	try
	{
		if (inference)
		{
			return infere(inputInferencePath);
		}
		else
		{
			auto validationDataset = parseInputDataset(validationFiles, cnn.getInputSize(), cnn.getOutputSize(), validationOffset, validationNum);
			auto trainingDataset = parseInputDataset(trainingFiles, cnn.getInputSize(), cnn.getOutputSize(), trainingOffset, trainingNum);
			
			auto exitCode = EXIT_SUCCESS;
			if (training)
			{
				exitCode = train(trainingDataset, trainingSettings, optimizer, lossFunction, validationDataset);
			}

			if (validation && !trainingSettings.periodicValidation)
			{
				if (exitCode == EXIT_SUCCESS)
				{
					exitCode = validate(validationDataset);
				}
				else
				{
					std::cerr << "Problems occured during training, skipping validation." << std::endl;
				}
			}

			return exitCode;
		}
	}
	catch (const CNNException & e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}


/*
 * @brief Inferes output based on given PNG input
 */
int CommandLineInterface::infere(const std::string & inputPath)
{
	auto image = PngParser::parseInputImage(inputPath, grayscale);

	cnn.run(image);

	return EXIT_SUCCESS;
}


/*
 * @brief Trains Convolutional Neural Network on given set of training data
 */
int CommandLineInterface::train(DatasetType trainingData, TrainingSettings & trainingSettings,
	std::shared_ptr<IOptimizer> optimizer, const LossFunctionType & lossFunctionType, const DatasetType validationData /*= {}*/)
{
	// Train the network on given dataset
	if (trainingData.empty())
	{
		std::cout << "No data to train on, dataset empty." << std::endl;
		return EXIT_FAILURE;
	}	

	// If best should be kept, set callback
	if (keepBest)
	{
		cnn.setOnEpochFinishedCallback(
			[this](unsigned, TrainingSettings &, float, float epochAccuracy, float)
			{
				this->keepBestCallback(epochAccuracy);
			}
		);
	}
		
	cnn.train(trainingSettings, trainingData, lossFunctionType, optimizer, validationData);

	// Dump network if user wanted it
	if (saveWeights && !keepBest)
	{
		return dumpNetworkToDisk();
	}

	return EXIT_SUCCESS;
}


/*
 * @brief Validates Convolutional Neural Network on given set of validation data
 */
int CommandLineInterface::validate(const DatasetType validationData)
{
	if (validationData.empty())
	{
		std::cout << "No data to validate on, dataset empty." << std::endl;
		return EXIT_FAILURE;
	}

	cnn.validate(validationData);

	return EXIT_SUCCESS;
}


/*
 * @brief Parses training data
 */
DatasetType CommandLineInterface::parseInputDataset(const std::vector<std::string> & files, const Dimensions inputSize, const Dimensions outputSize, const unsigned offset, const  unsigned toLoad)
{
	auto input = DatasetType();
	auto flattenedOutputSize = outputSize.width * outputSize.height * outputSize.depth;

	for (const auto & file : files)
	{
		// Load data based on their format
		if (std::regex_match(file.begin(), file.end(), std::regex(".+\\.[^\\.]*idx[^\\.]*$")))
		{
			input = IdxParser::parseLabelledImages(file, std::regex_replace(std::regex_replace(file, std::regex("images"), "labels"), std::regex("idx3"), "idx1"), 
				flattenedOutputSize, offset, toLoad);
		}
		else if (std::regex_match(file.begin(), file.end(), std::regex(".+\\.[^\\.]*bin[^\\.]*$")))
		{
			input = BinaryParser::parseLabelledImages(file, inputSize.width, inputSize.height, inputSize.depth, flattenedOutputSize, offset, toLoad);
		}
		else if (std::regex_match(file.begin(), file.end(), std::regex("^.+\\.txt$")))
		{
			input = PngParser::parseLabelledImages(file, flattenedOutputSize, grayscale, offset, toLoad);
		}
		else
		{
			std::cerr << "Input data file not detected as either BIN, IDX or TXT file (based on extension)." << std::endl;
		}
	}

	return input;
}


/*
 * @brief Dumps network to disk
 */
int CommandLineInterface::dumpNetworkToDisk()
{
	try
	{
		auto persistence = Persistence();
		persistence.dumpNetwork(cnn, cnnPath);
		return EXIT_SUCCESS;
	}
	catch (const std::exception & e)
	{
		std::cerr << "Could not save network to disk.\n  Reason: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
}


/*
 * @brief Keep best callback
 */
void CommandLineInterface::keepBestCallback(float epochAccuracy)
{
	static auto bestAccuracy = -1.0f;
	if (epochAccuracy > bestAccuracy)
	{
		bestAccuracy = epochAccuracy;
		dumpNetworkToDisk();
	}
}


/*
 * @brief Prints out error that occured when parsing arguments
 */
void CommandLineInterface::errorWhenParsingArguments(const std::string & reason)
{
	std::cerr << "Error when parsing arguments: " << reason << std::endl << "Use \"-h\" for help." << std::endl;
}