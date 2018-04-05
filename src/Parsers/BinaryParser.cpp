/*
 * @author Petr Rek
 * @project CNN Library
 * @brief IDX parser
 */

#include "src/Parsers/BinaryParser.h"

#include <vector>
#include <fstream>
#include <stdint.h>
#include <iostream>

/*
 * @brief Parses binary data format into labelled images
 */
std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> BinaryParser::parseLabelledImages(
	const std::string & path,
	const unsigned width,
	const unsigned height,
	const unsigned depth,
	const unsigned numberOfClasses,
	const unsigned skipFirstNum /*= 0*/,
	const unsigned maxParsedNum /*= 0*/,
	const float normalizationFactor /*= 255.0f*/)
{
	std::ifstream input(path, std::ios::binary | std::ios::ate);
	std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> output;

	if (input.is_open())
	{
		auto imageSize = width*height*depth + 1u;
		auto fileSize = input.tellg();
		input.seekg(0, std::ios::beg);

		auto parsedSize = 0u;
		auto parsedNum = 0u;

		auto buffer = new uint8_t[imageSize];

		while (parsedSize < fileSize)
		{
			if (maxParsedNum != 0 && parsedNum >= skipFirstNum + maxParsedNum)
			{
				break;
			}
			else if (parsedNum >= skipFirstNum)
			{
				unsigned char label;

				input.read((char *) &label, sizeof(label));

				auto img = Image<ForwardType>(Dimensions{ width, height, depth });
				auto cnt = 0u;
				for (auto k = 0u; k < img.getDepth(); k++)
				{
					for (auto j = 0u; j < img.getHeight(); j++)
					{
						for (auto i = 0u; i < img.getWidth(); i++)
						{
							unsigned char byte;
							input.read((char *) &byte, sizeof(byte));
							img(i, j, k) = static_cast<ForwardType>(byte / normalizationFactor);
							cnt++;
						}
					}
				}

				auto labelImg = createImageFromLabel(static_cast<unsigned>(label), numberOfClasses);

				output.emplace_back(img, labelImg);
			}

			parsedSize += imageSize;
			parsedNum++;
		}

		delete [] buffer;
	}

	return output;
}


/*
 * @brief Creates an image from label (in proper format)
 */
Image<ForwardType> BinaryParser::createImageFromLabel(const unsigned label, const unsigned numberOfClasses)
{
	std::vector<ForwardType> expected;

	for (auto i = 0u; i < numberOfClasses; i++)
	{
		if (i == label)
		{
			expected.push_back(1.0f);
		} 
		else
		{
			expected.push_back(0.0f);
		}
	}

	return expected;
}

