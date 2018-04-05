/*
 * @author Petr Rek
 * @project CNN library
 * @brief Demo example classifying grayscale images of digits on the run
 */

#include "src/ConvolutionalNeuralNetwork.h"
#include "src/Parsers/PngParser.h"
#include "src/Utils/Persistence.h"
#include "src/Utils/Limits.h"

#include <fstream>
#include <iomanip> 
#include <iostream>
#include <thread>

/*
 * @brief Produces CNN output to console
 */
void printOutput(const Image<ForwardType> & output)
{
	auto outputVector = output.getImageAsVector();

	// Output will be probabilities, we have to compute those
	auto sum = static_cast<ForwardType>(0);
	auto min = Limits::getMaximumValue<ForwardType>();
	auto max = Limits::getMinimumValue<ForwardType>();
	for (const auto val : outputVector)
	{
		if (val < min)
		{
			min = val;
		}
		if (val > max)
		{
			max = val;
		}
		sum += fabs(val);
	}

	// Output probabilities (pretty print)
	std::cout << "=================================================================================" << std::endl;
	std::cout << "|   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |" << std::endl;
	std::cout << "|";
	for (const auto val : outputVector)
	{
		auto probability = round((val - min) / sum * 10000) / 100; // round to two decimal places
		std::cout << " " << std::fixed << std::setprecision(1) << ((probability < 10) ? " " : "") << probability << "% |";
	}
	std::cout << "\n|";
	for (const auto val : outputVector)
	{
		if (val == max)
		{
			std::cout << " ***** |";
		}
		else
		{
			std::cout << "       |";
		}
	}
	std::cout << "\n=================================================================================\n\n" << std::endl;
}

/*
 * @brief Main method
 */
int main(int argc, char ** argv)
{
	// Check params
	if (argc != 3)
	{
		std::cerr << "Expecting learned XML file with CNN as first parameter and PNG file to scan as second one." << std::endl;
		return EXIT_FAILURE;
	}
	
	// Load CNN with weights
	ConvolutionalNeuralNetwork cnn;
	try
	{
		auto persistence = Persistence();
		cnn = persistence.loadNetwork(argv[1], true);
	}
	catch (const std::exception & e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	
	// Periodically check input file and recompute if change is encountered
	Image<ForwardType> inputImg;
	while (true)
	{
		// Load current file
		Image<ForwardType> tmp;
		try
		{
			tmp = PngParser::parseInputImage(argv[2], true);
		} 
		catch (const std::exception & e)
		{
			std::cerr << e.what() << std::endl;
			return EXIT_FAILURE;
		}

		// If changed, run it and produce output
		if (tmp != inputImg)
		{
			inputImg = tmp;
			if (inputImg.getDimensions() != cnn.getInputSize())
			{
				std::cerr << "Expecting image " << cnn.getInputSize().width << "x" << cnn.getInputSize().height << ". Try again." << std::endl;
			}
			else
			{
				auto output = cnn.run(inputImg);
				printOutput(output);
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}

	return EXIT_SUCCESS;
}