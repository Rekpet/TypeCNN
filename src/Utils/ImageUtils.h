/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Utils to work with Image class
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include "src/Image.h"

#include <exception>
#include <vector>
#include <string>

/*
 * @brief Output file could not be generated to file system
 */
class CouldNotCreateOutputFile : public std::exception
{
};

/*
 * @brief Depth chosen to be outputted is not valid
 */
class DepthOutOfRange : public std::exception
{
};

/*
 * @brief Image to be dumped does not have 3 dimensions (RGB)
 */
class ImageIsNotRGB : public std::exception
{
};

/*
 * @brief Utils for working with Image class
 */
namespace ImageUtils
{

	template <class Type>
	void dumpImageAsText(const Image<Type> & img, const unsigned d = 0, const unsigned precision = 5, const Type normalizationFactor = 1.0f);

	template <class Type>
	void dumpColorImage(const Image<Type> & img, const std::string & path, const Type normalizationFactor = 255.0f);

	template <class Type>
	void dumpGrayscaleImage(const Image<Type> & img, const std::string & path, const unsigned d = 0, const Type normalizationFactor = 255.0f);

	template <class Type>
	Image<Type> normalizeImage(const Image<Type> & input);

	void dumpFiltersAsImages(const std::string & filePrefix, const std::vector<Image<BackwardType>> & filters, const std::vector<BackwardType> & biases);

} // namespace ImageUtils

#endif