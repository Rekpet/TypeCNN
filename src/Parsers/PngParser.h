/*
 * @author Petr Rek
 * @project CNN Library
 * @brief PNG parser
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef PNG_PARSER_H
#define PNG_PARSER_H

#include "src/Image.h"

#include <exception>

 /*
  * @brief Thrown if single image could not be opened
  */
class CouldNotOpenImage : public std::exception 
{
};


/*
 * @brief Thrown if images loaded in batches do not have the same dimensions
 */
class ImagesAreNotConsistent : public std::exception
{
};


/*
 * @brief Parses PNG images (single or multiple ones given by descriptor file
 */
class PngParser
{

public:

	static std::vector<std::pair<Image<ForwardType>, Image<ForwardType>>> parseLabelledImages(
		const std::string & descriptorPath,
		const unsigned classesNum,
		const bool grayscale,
		const unsigned skipFirstNum = 0,
		const unsigned maxParsedNum = 0,
		const float normalizationFactor = 255.0f);

	static Image<ForwardType> parseInputImage(
		const std::string & path,
		const bool grayscale,
		const float normalizationFactor = 255.0f);

private:

	static bool fileExists(const std::string & path);

	static std::pair<std::string, std::vector<ForwardType>> parseDescriptorFileLine(const std::string & line);

};

#endif
