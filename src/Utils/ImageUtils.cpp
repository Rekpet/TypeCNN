/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Utils to work with Image class
 */

#include "src/Utils/ImageUtils.h"

#include "src/Image.h"
#include "src/Utils/Limits.h"

#include "3rdParty/LodePNG/lodepng.h"

#include <exception>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

/*
 * @brief Utils for working with Image class
 */
namespace ImageUtils
{

	/*
	 * @brief Writes given depth of image to std::out
	 */
	template <class Type>
	void dumpImageAsText(const Image<Type> & img, const unsigned d /*= 0*/, const unsigned precision /*= 5*/, const float normalizationFactor /*= 1.0f*/)
	{
		if (d >= img.getDepth())
		{
			throw DepthOutOfRange();
		}
		else
		{
			std::cout << std::endl;

			for (auto i = 0u; i < img.getHeight(); i++)
			{
				for (auto j = 0u; j < img.getWidth(); j++)
				{
					auto strVal = to_string(img(j, i, d) * normalizationFactor);
					auto strLength = strVal.size();
					if (strLength > precision)
					{
						strVal = strVal.substr(0, precision);
						if (strVal[precision - 1] == '.')
						{
							strVal[precision - 1] = ' ';
						}
					}
					else if (strLength < precision)
					{
						for (auto k = 0u; k < (precision - strLength); k++)
						{
							strVal.append(" ");
						}
					}
					std::cout << strVal << " ";
				}
				std::cout << std::endl;
			}

			std::cout << std::endl;
		}
	}


	/*
	 * @brief Dumps given 3D image as color PNG image (R, G, B)
	 */
	template <class Type>
	void dumpColorImage(const Image<Type> & img, const std::string & path, const Type normalizationFactor /*= 255.0f*/)
	{
		if (img.getDepth() != 3)
		{
			throw ImageIsNotRGB();
		}
		else
		{
			std::vector<unsigned char> out;

			for (auto i = 0u; i < img.getHeight(); i++)
			{
				for (auto j = 0u; j < img.getWidth(); j++)
				{
					// workaround because of fixed point type
					out.push_back(static_cast<char>(std::stoi(std::to_string(img(j, i, 0) * normalizationFactor)))); // R
					out.push_back(static_cast<char>(std::stoi(std::to_string(img(j, i, 1) * normalizationFactor)))); // G
					out.push_back(static_cast<char>(std::stoi(std::to_string(img(j, i, 2) * normalizationFactor)))); // B
					out.push_back(255);                                                                              // A
				}
			}

			auto error = lodepng::encode(path, out, img.getWidth(), img.getHeight());

			if (error)
			{
				throw CouldNotCreateOutputFile();
			}
		}
	}

	/*
	 * @brief Dumps given depth of image as grayscale PNG image
	 */
	template <class Type>
	void dumpGrayscaleImage(const Image<Type> & img, const std::string & path, const unsigned d /*= 0*/, const Type normalizationFactor /*= 255.0f*/)
	{
		if (d >= img.getDepth())
		{
			throw DepthOutOfRange();
		}
		else
		{
			std::vector<unsigned char> out;

			for (auto i = 0u; i < img.getHeight(); i++)
			{
				for (auto j = 0u; j < img.getWidth(); j++)
				{
					auto pixelVal = static_cast<char>(std::stoi(std::to_string(img(j, i, d) * normalizationFactor)));
					out.push_back(pixelVal); // R
					out.push_back(pixelVal); // G
					out.push_back(pixelVal); // B
					out.push_back(255);      // A
				}
			}

			auto error = lodepng::encode(path, out, img.getWidth(), img.getHeight());

			if (error)
			{
				throw CouldNotCreateOutputFile();
			}
		}
	}


	/*
	 * @brief Normalizes content of matrix to range <0, 1>
	 */
	template <class Type>
	Image<Type> normalizeImage(const Image<Type> & input)
	{
		auto output = Image<Type>(input.getDimensions());
		auto flattenedSize = input.getFlattenedSize();

		auto min = static_cast<Type>(Limits::getMaximumValue<Type>());
		auto max = static_cast<Type>(Limits::getMinimumValue<Type>());
		for (auto i = 0u; i < flattenedSize; i++)
		{
			if (input(i) < min)
			{
				min = input(i);
			}
			if (input(i) > max)
			{
				max = input(i);
			}
		}

		for (auto i = 0u; i < flattenedSize; i++)
		{
			output(i) = (input(i) - min) / max;
		}

		return output;
	}


	/*
	 * @brief Dumps filters (only support grayscale or RGB)
	 */
	void dumpFilters(const std::string & filePrefix, const std::vector<Image<BackwardType>> & filters, const std::vector<BackwardType> & biases)
	{
		auto cnt = 0u;
		for (const auto & filter : filters)
		{
			if (filter.getDepth() == 1u)
			{
				dumpGrayscaleImage<BackwardType>(normalizeImage<BackwardType>(filter), filePrefix + "_" + std::to_string(++cnt), 0, 255.0f);
			}
			else if (filter.getDepth() == 3u)
			{
				dumpColorImage<BackwardType>(normalizeImage<BackwardType>(filter), filePrefix + "_" + std::to_string(++cnt), 255.0f);
			}
			else
			{
				throw DepthOutOfRange();
			}
		}

		std::ofstream biasOutput(filePrefix + "_biases.txt");
		if (biasOutput.is_open())
		{
			for (const auto & val : biases)
			{
				biasOutput << val << "\n";
			}
		}
		else
		{
			throw CouldNotCreateOutputFile();
		}
	}

} // namespace ImageUtils