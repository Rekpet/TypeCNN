/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Matrix containing all the data
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef IMAGE_H
#define IMAGE_H

#include "src/CompileSettings.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/*
 * @brief Coordinates of point in 3D system
 */
struct Point
{

	unsigned width;

	unsigned height;

	unsigned depth;

	bool operator==(const Point & d)
	{
		return (width == d.width) && (height == d.height && depth == d.depth);
	}

	bool operator!=(const Point & d)
	{
		return !(*this == d);
	}

};

/*
 * @brief Describes dimensions of matrix
 */
struct Dimensions
{

	unsigned width;

	unsigned height;

	unsigned depth;

	bool operator==(const Dimensions & d)
	{
		return (width == d.width) && (height == d.height && depth == d.depth);
	}

	bool operator!=(const Dimensions & d)
	{
		return !(*this == d);
	}

};

/*
 * @brief Input/Output of layers
 */
template <typename TYPE>
class Image
{

/*
	Matrix is saved row wise, third coordinate being depth

	Coordinate convention:
		Based on other implementation and maths behind CNN, we use this coordinate system

		1 2     Element 2 has coordinates (1, 0, j)
		3 4                               (x, y, z)
		           Where j is depth (third dimension)
				   Thus x = columns, y = rows, z = depth

		Since this would be hard to describe in tests, tests use completely opposite approach,
		    thus this class offers method to load data from user readable format
 */

public:

	/*
	 * @brief Creates empty image
	 */
	Image()
		: dimensions(Dimensions{ 0, 0, 0 })
		, flattenedSize(0)
	{
		image = nullptr;
	}


	/*
	 * @brief Creates new empty image with given sizes
	 */
	Image(const Dimensions dimensions)
		: dimensions(dimensions)
		, flattenedSize(dimensions.width * dimensions.height * dimensions.depth)
	{
		image = std::shared_ptr<TYPE>(new TYPE[flattenedSize], std::default_delete<TYPE[]>());
	}


	/* 
	 * @brief Creates image from user readable representation (different coordinate system)
	 */
	Image(const std::vector<std::vector<std::vector<TYPE>>> & img)
	{
		dimensions.width = img[0][0].size();
		dimensions.height = img[0].size();
		dimensions.depth = img.size();

		flattenedSize = dimensions.width * dimensions.height * dimensions.depth;
		image = std::shared_ptr<TYPE>(new TYPE[flattenedSize], std::default_delete<TYPE[]>());

		for (auto z = 0u; z < dimensions.depth; z++)
		{
			for (auto y = 0u; y < dimensions.height; y++)
			{
				for (auto x = 0u; x < dimensions.width; x++)
				{
					image.get()[z * dimensions.height * dimensions.width + y * dimensions.width + x] = img[z][y][x];
				}
			}
		}
	}


	/*
	 * @brief Creates image from 1D vector
	 */
	Image(const std::vector<TYPE> & out)
		: Image(Dimensions{ static_cast<unsigned>(out.size()), 1, 1 })
	{
		int cnt = 0;
		for (auto & val : out)
		{
			image.get()[cnt] = val;
			cnt++;
		}
	}


	/*
	 * @brief Equality operator (does not account for floating point mismatch!)
	 */
	bool operator==(const Image & other)
	{
		if (dimensions != other.dimensions)
		{
			return false;
		}

		for (auto i = 0u; i < flattenedSize; i++)
		{
			if (image.get()[i] != other(i))
			{
				return false;
			}
		}

		return true;
	}


	/*
	 * @brief Inequality operator
	 */
	bool operator!=(const Image & other)
	{
		return !(*this == other);
	}


	/*
	 * @brief Assignment operator, needs to realloc memory
	 */
	Image & operator=(const Image & other)
	{
		dimensions = other.dimensions;
		flattenedSize = other.flattenedSize;
		image = std::shared_ptr<TYPE>(new TYPE[flattenedSize], std::default_delete<TYPE[]>());
		std::memcpy(image.get(), other.image.get(), flattenedSize * sizeof(TYPE));
		return *this;
	}


	/*
	 * @brief Returns output as simple vector
	 */
	std::vector<TYPE> getImageAsVector() const
	{
		std::vector<TYPE> out;
		out.resize(flattenedSize);
		for (auto i = 0u; i < flattenedSize; i++)
		{
			out[i] = image.get()[i];
		}
		return out;
	}


	/*
	 * @brief Fills the entire image with zeros
	 */
	void clear()
	{
		memset(image.get(), 0, flattenedSize * sizeof(*image.get()));
	}


	/*
	 * @brief Returns size of flattened image (stored that way)
	 */
	inline unsigned getFlattenedSize() const
	{
		return flattenedSize;
	}


	/*
	 * @brief Returns dimensions of matrix
	 */
	inline Dimensions getDimensions() const
	{
		return dimensions;
	}


	/*
	 * @brief Returns depth
	 */
	inline unsigned getDepth() const
	{
		return dimensions.depth;
	}


	/*
	 * @brief Returns height
	 */
	inline unsigned getHeight() const
	{
		return dimensions.height;
	}


	/*
	 * @brief Returns width
	 */
	inline unsigned getWidth() const
	{
		return dimensions.width;
	}


	/*
	 * @brief Returns pixel reference at given 3D coordinates (not implementation dependent)
	 */
	inline TYPE & operator() (const unsigned & x, const unsigned & y, const unsigned & z) const
	{
		return image.get()[z * dimensions.height * dimensions.width + y * dimensions.width + x];
	}


	/*
	 * @brief Returns pixel reference at given 2D coordinates (not implementation dependent)
	 */
	inline TYPE & operator() (const unsigned & x, const unsigned & y) const
	{
		return image.get()[y * dimensions.width + x];
	}


	/*
	 * @briefReturns pixel reference at given offset (accessing as flattened)
	 */
	inline TYPE & operator() (const unsigned & x) const
	{
		return image.get()[x];
	}

private:

	/// Image itself in shared_ptr for automatic deletion
	std::shared_ptr<TYPE> image;

	/// Dimensions
	Dimensions dimensions;

	/// Flattened linearized size
	unsigned flattenedSize;

};

#endif