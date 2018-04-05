/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Binary parser
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef BINARY_PARSER_H
#define BINARY_PARSER_H

#include "src/Image.h"

/*
 File format:
		8 bit unsigned = label
		width * height * depth * 8 bit unsigned = pixels

		Stored row by row, no delimiters

		e.g. CIFAR-10
		First byte is label, then 3072 pixels - 32*32*3 (in order R, G, B)
 */

/*
 * @brief Parses binary data format into labelled images
 */
class BinaryParser
{

public:

	static std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> parseLabelledImages(
		const std::string & path,
		const unsigned width,
		const unsigned height,
		const unsigned depth,
		const unsigned numberOfClasses,
		const unsigned skipFirstNum = 0,
		const unsigned maxParsedNum = 0,
		const float normalizationFactor = 255.0f);

private:

	static Image<ForwardType> createImageFromLabel(const unsigned label, const unsigned numberOfClasses);

};

#endif