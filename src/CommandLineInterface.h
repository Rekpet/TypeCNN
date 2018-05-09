/*
 * @author Petr Rek
 * @project Convolutional Neural Network library
 * @brief CNN console interface
 */

#include "src/Image.h"
#include "src/ConvolutionalNeuralNetwork.h"

#include "3rdParty/CxxOpts/cxxopts.hpp"

#include <string>
#include <vector>

using DatasetType = std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>>;

/*
 * @brief Command line interface for TypeCNN (uses standard output !!!)
 */
class CommandLineInterface
{

public:

	CommandLineInterface();

	int runWithGivenArguments(int argc, char ** argv);

private:

	int infere(const std::string & inputPath);


	int train(DatasetType trainingData, TrainingSettings & trainingSettings, std::shared_ptr<IOptimizer> optimizer, 
		const LossFunctionType & lossFunctionType, const DatasetType validationData = {});

	int validate(const DatasetType validationData);

	DatasetType parseInputDataset(const std::vector<std::string> & files, 
		const Dimensions inputSize, const Dimensions outputSize, const unsigned offset, const  unsigned toLoad);

	int dumpNetworkToDisk();

	void keepBestCallback(float epochAccuracy);

	void errorWhenParsingArguments(const std::string & reason);

	void showTypeInfo();

private:

	/// Convolutional neural network to work with
	ConvolutionalNeuralNetwork cnn;

	/// Path on disk to CNN
	std::string cnnPath;

	/// Save weight after training?
	bool saveWeights = true;

	/// Keep best solution after each epoch?
	bool keepBest = false;

	/// Grayscale? (for when loading PNG files)
	bool grayscale = false;

	/// Argument parser
	cxxopts::Options options;

};
