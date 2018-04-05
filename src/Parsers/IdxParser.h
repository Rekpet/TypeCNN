/*
 * @author Petr Rek
 * @project CNN Library
 * @brief IDX parser
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef IDX_PARSER_H
#define IDX_PARSER_H

#include "src/Image.h"

/*
 File format:
	Label file:
		32 bit integer  = magic number
		32 bit integer  = number of items
		8  bit unsigned = label
		...
		8  bit unsigned = label

	Image file:
		32 bit integer  = magic number
		32 bit integer  = number of items
		32 bit integer  = number of rows
		32 bit integer  = number of columns
		8  bit unsigned = pixel value
		...
		8  bit unsigned = pixel value
 */

/*
 * @brief Parser for two-dimensional IDX format
 */
class IdxParser
{

public:

	static std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> parseLabelledImages(
		const std::string & imagesPath, 
		const std::string & labelsPath,
		const unsigned classesNum,
		const unsigned skipFirstNum = 0,
		const unsigned maxParsedNum = 0,
		const float normalizationFactor = 255.0f);

private:

	static std::vector<Image<ForwardType>> readLabels(
		const std::string & labelPath, 
		const unsigned classesNum, 
		const unsigned skipFirstNum, 
		const unsigned maxParsedNum);

	static std::vector<Image<ForwardType>> readImages(
		const std::string & imagePath, 
		const unsigned skipFirstNum, 
		const unsigned maxParsedNum,
		const float normalizationFactor);

	static uint32_t convertHeighEndianToLittleEndian(
		const uint32_t val);

};

#endif