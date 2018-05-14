/*
 * @author Petr Rek
 * @project CNN Library
 * @brief PNG parser
 */

#include "src/Parsers/PngParser.h"

#include "3rdParty/LodePNG/lodepng.h"

#include <fstream>
#include <sstream>
#include <stdint.h>

/*
 * @brief Parses single PNG image
 */
Image<ForwardType> PngParser::parseInputImage(const std::string & path, const bool grayscale, const float normalizationFactor /*= 255.0f*/)
{
	std::vector<unsigned char> image;
	unsigned width, height;

	if (lodepng::decode(image, width, height, path))
	{
		throw CouldNotOpenImage("PNG file could not be opened.");
	}

	Dimensions imgDim = { width, height, grayscale ? 1u : 3u };

	Image<ForwardType> img(imgDim);
	
	auto cnt = 0u;
	for (auto i = 0u; i < img.getHeight(); i++)
	{
		for (auto j = 0u; j < img.getWidth(); j++)
		{
			auto r = static_cast<ForwardType>(image[cnt++] / normalizationFactor); // R
			auto g = static_cast<ForwardType>(image[cnt++] / normalizationFactor); // G
			auto b = static_cast<ForwardType>(image[cnt++] / normalizationFactor); // B
			cnt++; // A

			if (grayscale)
			{
				img(j, i, 0) = r;
			}
			else
			{
				img(j, i, 0) = r;
				img(j, i, 1) = g;
				img(j, i, 2) = b;
			}
		}
	}

	return img;
}


/*
 * @brief Parses mutiple PNG images described in a text file
 */
std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> PngParser::parseLabelledImages(
	const std::string & descriptorPath, const unsigned classesNum, const bool grayscale, const unsigned skipFirstNum /*= 0*/, const unsigned maxParsedNum /*= 0*/, const float normalizationFactor /*= 255.0f*/)
{
	std::ifstream input(descriptorPath);
	std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> out;

	if (input.is_open())
	{
		size_t found;
		found = descriptorPath.find_last_of("/\\");
		auto rootPath = descriptorPath.substr(0, found);

		auto cnt = 0u;
		for (std::string line; std::getline(input, line); )
		{
			if (cnt >= skipFirstNum && (maxParsedNum == 0 || cnt < skipFirstNum + maxParsedNum))
			{	
				auto parsedLine = parseDescriptorFileLine(line);
				auto imagePath = rootPath + "/" + parsedLine.first;

				if (parsedLine.second.size() == classesNum && fileExists(imagePath))
				{
					auto img = parseInputImage(imagePath, grayscale, normalizationFactor);

					if (!out.empty() && out.front().first.getDimensions() != img.getDimensions())
					{
						throw ImagesAreNotConsistent("Image sizes are not consistent.");
					}

					out.emplace_back(img, parsedLine.second);
				}
				else
				{
					throw ImagesAreNotConsistent("Image sizes are not consistent.");
				}
			}

			cnt++;
		}
	}

	return out;
}


/*
 * @brief Checks that file exists
 */
bool PngParser::fileExists(const std::string & path)
{
	std::ifstream infile(path);
	return infile.good();
}


/*
 * @brief Parses line of a descriptor file
 */
std::pair<std::string, std::vector<ForwardType>> PngParser::parseDescriptorFileLine(const std::string & line)
{
	std::stringstream ss(line);

	std::string path;
	std::vector<ForwardType> outputValues;

	ss >> path;

	ForwardType buffer;
	while (ss >> buffer)
	{
		outputValues.push_back(buffer);
	}

	return { path, outputValues };
}