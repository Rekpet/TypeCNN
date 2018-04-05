/*
 * @author Petr Rek
 * @project Convolutional Neural Network library
 * @brief CNN console interface and sample usage
 */

#include "src/CommandLineInterface.h"

/*
 * @brief Main method serving as CLI and sample usage of library
 */
int main(int argc, char ** argv)
{
	auto cli = CommandLineInterface();
	return cli.runWithGivenArguments(argc, argv);
}