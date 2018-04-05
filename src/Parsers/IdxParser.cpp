/*
 * @author Petr Rek
 * @project CNN Library
 * @brief IDX parser
 */

#include "src/Parsers/IdxParser.h"

#include <vector>
#include <fstream>
#include <stdint.h>
#include <iostream>

/*
 * @brief Parses labelled images of IDX format
 */
std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> IdxParser::parseLabelledImages(
	const std::string & imagesPath, const std::string & labelsPath, 
	const unsigned classesNum, const unsigned skipFirstNum /*= 0*/, 
	const unsigned maxParsedNum /*= 0*/,
	const float normalizationFactor /*= 255.0f*/)
{
	auto images = readImages(imagesPath, skipFirstNum, maxParsedNum, normalizationFactor);
	auto labels = readLabels(labelsPath, classesNum, skipFirstNum, maxParsedNum);

	std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> out;

	auto labelIt = labels.begin();
	auto imageIt = images.begin();
	for ( /* Above, so we can use auto */ ;
		labelIt != labels.end() && imageIt != images.end();
		labelIt++, imageIt++)
	{
		out.push_back({std::move(*imageIt), std::move(*labelIt)});
	}

	return out;
}


/*
 * @brief Parses labeles from file
 */
std::vector<Image<ForwardType>> IdxParser::readLabels(const std::string & labelPath, const unsigned classesNum, 
	const unsigned skipFirstNum, const unsigned maxParsedNum)
{
	std::ifstream input(labelPath);
	std::vector<Image<ForwardType>> out;

	if (input.is_open())
	{
		uint32_t magicNumber;
		uint32_t itemsNum;

		input.read((char *)(&magicNumber), sizeof(magicNumber));
		input.read((char *)(&itemsNum), sizeof(itemsNum));

		itemsNum = convertHeighEndianToLittleEndian(itemsNum);

		if (skipFirstNum > itemsNum)
		{
			return out;
		}
		else if (itemsNum > (maxParsedNum + skipFirstNum) && maxParsedNum != 0)
		{
			itemsNum = maxParsedNum + skipFirstNum;
		}

		for (auto i = 0u; i < itemsNum; i++)
		{
			std::vector<ForwardType> expected;
			uint8_t byte;
			input.read((char *)(&byte), sizeof(byte));
			if (i >= skipFirstNum)
			{
				auto byteVal = static_cast<unsigned>(byte);
				for (auto j = 0u; j < classesNum; j++)
				{
					if (j == byteVal)
					{
						expected.push_back(1.0f);
					} 
					else
					{
						expected.push_back(0.0f);
					}
				}
				out.push_back(expected);
			}
		}
	}

	return out;
}


/*
 * @brief Parses images from file
 */
std::vector<Image<ForwardType>> IdxParser::readImages(const std::string & imagePath,
	const unsigned skipFirstNum, const unsigned maxParsedNum, const float normalizationFactor)
{
	std::ifstream input(imagePath, std::ios::binary);
	std::vector<Image<ForwardType>> out;

	if (input.is_open())
	{
		uint32_t magicNumber;
		uint32_t itemsNum;
		uint32_t rowsNum;
		uint32_t columnsNum; 

		input.read((char *)(&magicNumber), sizeof(magicNumber));
		input.read((char *)(&itemsNum), sizeof(itemsNum));
		input.read((char *)(&rowsNum), sizeof(rowsNum));
		input.read((char *)(&columnsNum), sizeof(columnsNum));

		itemsNum = convertHeighEndianToLittleEndian(itemsNum);
		rowsNum = convertHeighEndianToLittleEndian(rowsNum);
		columnsNum = convertHeighEndianToLittleEndian(columnsNum);

		Dimensions imageSize;
		imageSize.height = rowsNum;
		imageSize.width = columnsNum;
		imageSize.depth = 1;

		if (skipFirstNum > itemsNum)
		{
			return out;
		}
		else if (itemsNum > (maxParsedNum + skipFirstNum) && maxParsedNum != 0)
		{
			itemsNum = maxParsedNum + skipFirstNum;
		}

		uint8_t * imgData = new uint8_t[imageSize.height * imageSize.width];
		for (auto i = 0u; i < itemsNum; i++)
		{
			Image<ForwardType> image(imageSize);

			input.read((char *)imgData, imageSize.height * imageSize.width);

			if (i >= skipFirstNum)
			{	
				for (auto r = 0u; r < imageSize.height; r++)
				{
					for (auto c = 0u; c < imageSize.width; c++)
					{
						image(c, r, 0) = static_cast<ForwardType>(imgData[c + imageSize.width * r] / normalizationFactor);
					}
				}

				out.push_back(std::move(image));
			}
		}
		delete [] imgData;
	}

	return out;
}


/*
 * @brief Converts high endian to little endian
 */
uint32_t IdxParser::convertHeighEndianToLittleEndian(const uint32_t val)
{
	return (val >> 24) |
		   ((val << 8) & 0x00FF0000) |
		   ((val >> 8) & 0x0000FF00) |
		   (val << 24);
}