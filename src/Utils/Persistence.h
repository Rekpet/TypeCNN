/*
 * @author Petr Rek
 * @project CNN Library
 * @brief Persistence module for CNN
 */

#ifdef _MSC_VER
#pragma once
#endif

#ifndef PERSISTENCE_H
#define PERSISTENCE_H

#include "src/ConvolutionalNeuralNetwork.h"

#include "src/LayerAliases.h"

#include "3rdParty/TinyXML2/tinyxml2.h"

#include <exception>

/*
 * @brief Universal exception that can be thrown when parsing/dumping CNN
 */
class PersistenceException : public std::runtime_error
{
	public:
		explicit PersistenceException(const std::string & msg) 
			: std::runtime_error(msg.c_str()) 
		{
		};
};

/*
 * @brief Thrown if weights could not be loaded due to different sizes
 */
class InvalidWeights : public PersistenceException 
{ 
	using PersistenceException::PersistenceException; 
};

/*
 * @brief Thrown if filters could not be loaded due to inconsistent dimensions
 */
class InvalidFilters : public PersistenceException 
{ 
	using PersistenceException::PersistenceException; 
};

/*
 * @brief Thrown if input XML file was not valid CNN description
 */
class CouldNotParseXmlFile : public PersistenceException 
{
	using PersistenceException::PersistenceException; 
};

/*
 * @brief Thrown if CNN could not be dumped to file system
 */
class CannotCreateFilesOnDisk : public PersistenceException 
{
	using PersistenceException::PersistenceException; 
};

/*
 * @brief Thrown if convolutional neural network is not valid (reason provided)
 */
class InvalidConvolutionalNeuralNetwork : public PersistenceException 
{ 
	using PersistenceException::PersistenceException; 
};

/*
 * @brief Contains settings that were parsed from XML file
 */
struct ParsedSettings
{

	TaskType taskType;

	LossFunctionType lossFunctionType;

	Dimensions input;

	Dimensions output;

};

/*
 * @brief Class responsible for saving/loading state of CNN
 */
class Persistence
{

public:

	void dumpNetwork(const ConvolutionalNeuralNetwork & cnn, const std::string & pathToXmlFile);

	ConvolutionalNeuralNetwork loadNetwork(const std::string & pathToXmlFile, const bool loadWeigts);

	template <class OutType>
	void dumpWeights(const std::string & pathToWeights, const Image<BackwardType> & weights);

	template <class OutType>
	void dumpFilters(const std::string & pathToFilters, const std::vector<Image<BackwardType>> & filters, const std::vector<BackwardType> & biases);

	Image<BackwardType> parseWeights(const std::string & pathToWeights, const unsigned inputNeurons, const unsigned outputNeurons);

	std::pair<std::vector<Image<BackwardType>>, std::vector<BackwardType>> parseFilters(const std::string & pathToFilters,
		const unsigned & filterNum, const unsigned & extent, const unsigned & inputDepth);

private:

	void parseSettings(tinyxml2::XMLElement * settingsRoot);

	ConvolutionalNeuralNetwork parseArchitecture(tinyxml2::XMLElement * architectureRoot);

	std::shared_ptr<ILayer<ForwardType, WeightType>> parseConvolutionalLayer(tinyxml2::XMLElement * root, std::shared_ptr<ILayer<ForwardType, WeightType>> prevLayer);

	std::shared_ptr<ILayer<ForwardType, WeightType>> parsePoolingLayer(tinyxml2::XMLElement * root, std::shared_ptr<ILayer<ForwardType, WeightType>> prevLayer);

	std::shared_ptr<ILayer<ForwardType, WeightType>> parseDropoutLayer(tinyxml2::XMLElement * root, std::shared_ptr<ILayer<ForwardType, WeightType>> prevLayer);

	std::shared_ptr<ILayer<ForwardType, WeightType>> parseFullyConnectedLayer(tinyxml2::XMLElement * root, std::shared_ptr<ILayer<ForwardType, WeightType>> prevLayer);

	std::shared_ptr<ILayer<ForwardType, WeightType>> parseActivationLayer(tinyxml2::XMLElement * root, std::shared_ptr<ILayer<ForwardType, WeightType>> prevLayer);

	std::vector<std::string> splitLineByDelimiter(const std::string & line, const char & delimiter);



	void dumpSettings(tinyxml2::XMLNode * settingsRoot, tinyxml2::XMLDocument & document, const ConvolutionalNeuralNetwork & cnn);

	void dumpArchitecture(tinyxml2::XMLNode * architectureRoot, tinyxml2::XMLDocument & document, const ConvolutionalNeuralNetwork & cnn);

	void dumpConvolutionalLayer(tinyxml2::XMLNode * layerRoot, tinyxml2::XMLDocument & document, ConvolutionalLayer<ForwardType, WeightType> * layer);

	void dumpPoolingLayer(tinyxml2::XMLNode * layerRoot, tinyxml2::XMLDocument & document, PoolingLayer<ForwardType, WeightType> * layer);

	void dumpDropoutLayer(tinyxml2::XMLNode * layerRoot, tinyxml2::XMLDocument & document, DropoutLayer<ForwardType, WeightType> * layer);

	void dumpFullyConnectedLayer(tinyxml2::XMLNode * layerRoot, tinyxml2::XMLDocument & document, FullyConnectedLayer<ForwardType, WeightType> * layer);

	void dumpActivationLayer(tinyxml2::XMLNode * layerRoot, tinyxml2::XMLDocument & document, ActivationLayer<ForwardType, WeightType> * layer);

private:

	/// Specifies if weights should be loaded
	bool loadWeights = true;
	
	/// Specifies indes of layers that are dumped (to create file names)
	unsigned layerDumpIndex = 0;

	/// Directory in which we are operating
	std::string directory;

	/// Parsed settings
	ParsedSettings settings;

};

#endif
