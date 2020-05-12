/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/

/**
 * @file ldbc_import.h
 * @brief this class reads the ldbc files and generates the graph in CSR or AdjList
 * @todo support for array properties (for simplicity only last one take currently)
*/

#ifndef MORPHSTORE_LDBC_IMPORT_H
#define MORPHSTORE_LDBC_IMPORT_H

#include <core/storage/graph/graph.h>
#include "ldbc_schema.h"

#include <filesystem>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <regex>
#include <variant>
#include <chrono>
#include <string>
#include <stdexcept>
#include <optional>



// hash function used to hash a pair of any kind using XOR (for verticesMap)
struct hash_pair {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const
    {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ hash2;
    }
};

namespace morphstore{

    class LDBCImport {

    private:
        std::filesystem::path base_directory;
        std::vector<std::filesystem::path> verticesPaths;
        std::vector<std::filesystem::path> edgesPaths;
        std::map<unsigned short int, std::string> vertexTypeLookup;
        std::map<unsigned short int, std::string> edgeTypeLookup;
        // data structure for lookup local ids with vertexType to global system id: (vertexType, ldbc_id) -> global id
        std::unordered_map<std::pair<std::string, std::string>, uint64_t, hash_pair> globalIdLookupMap;

        // unordered_map for lookup system-id and its in the graph (for further processing, e.g. filling the edge_array in the right order)
        std::unordered_map<uint64_t, std::vector<morphstore::Edge>> vertexEdgesLookup;
        std::unordered_map<uint64_t,  std::unordered_map<std::string, property_type>> edgeProperties;

    public:
        // directory including a static/ and dynamic/ directory like in /ldbc_snb_datagen/social_network/
        LDBCImport(const std::string &dir) {
            base_directory = dir;
            insert_file_names();
        }

        std::string getDirectory() const {
            return base_directory;
        }

        // get the vertex or edge type based on the fileName
        std::string getEntityType(std::filesystem::path filePath) {
                // last [a-zA-Z] to remove ending _
                std::regex typeRegExp("[a-zA-Z_]+[a-zA-Z]");
                std::smatch match;

                std::string fileName = filePath.filename().string();

                if(std::regex_search(fileName, match, typeRegExp)) {
                    //std::cout << "EntityType: " << match[0] << std::endl;
                    //std::cout.flush();
                    return match[0];
                }
                else {
                    throw std::invalid_argument("No EntityType in: " + fileName);
                }
        }
        

        // function which iterates through the base_directory to receive file names (entire path)
        void insert_file_names() {
            
            std::filesystem::path dynamic_data_dir (base_directory / "dynamic"); 
            std::filesystem::path static_data_dir (base_directory / "static"); 
            std::vector<std::filesystem::path> dirs{dynamic_data_dir, static_data_dir}; 

            for(const auto dir: dirs) {
                for (const auto &entry : std::filesystem::directory_iterator(dir)) {
                    // ignore files starting with a '.' (+ 1 as '/' is the first character otherwise)
                    if (entry.path().string()[dir.u8string().length() + 1] == '.') {
                        continue;
                    } else {
                        // insert file path to vertices or edges vector
                        differentiate(entry.path().string());
                    }
                }
            }

            if(verticesPaths.empty()) {
                print_file_names();
                throw std::invalid_argument("No vertex files found");
            }
        }

        // this function differentiates, whether the file is a vertex or edge and puts it into the specific vector
        void differentiate(std::filesystem::path path) {
            // if the string contains a '_' -> it's a edge file; otherwise a vertex file
            // if string contains word_word it is an edge files (vertex files only contain one word)

            // a vertex file name contains exactly one word and after that only numbers are allowed f.i. _0_0
            // .*\\/ for path marks the directory path
            std::regex vertexFileRegExp("^(.*\\/)([a-zA-Z]+\\_)([0-9_]*).csv$");

            if (std::regex_match(path.u8string(), vertexFileRegExp)) {
                verticesPaths.push_back(path);
            } else {
                edgesPaths.push_back(path);
            }
        }

        // this function reads the vertices-files and creates vertices in a graph
        // + creates the vertexTypeLookup (number to string) for the graph
        void generate_vertices(Graph& graph) {
            std::cout << "(1/2) Generating LDBC-Vertices ...";
            std::cout.flush();

            // iterate through vector of vertex-addresses
            for (const auto &file : verticesPaths)
            {
                // data structure for attributes of entity, e.g. taglass -> id, name, url
                std::vector<std::pair<std::string, Ldbc_Data_Type>> attributes;

                std::string vertexType = getEntityType(file);
                int vertexTypeNumber = get_vertex_type_number(vertexType).value();

                char *buffer;

                uint64_t fileSize = 0;

                std::string address = file;

                std::ifstream vertexFile(address, std::ios::binary |
                                                      std::ios::ate); // 'ate' means: open and seek to end immediately after opening

                if (!vertexFile) {
                    std::cerr << "Error, opening file. ";
                    exit(EXIT_FAILURE);
                }

                // calculate file size
                if (vertexFile.is_open()) {
                    // tellg() returns: The current position of the get pointer in the stream on success, pos_type(-1) on failure.
                    fileSize = static_cast<uint64_t>(vertexFile.tellg()); 
                    vertexFile.clear();
                    // Seeks to the very beginning of the file, clearing any fail bits first (such as the end-of-file bit)
                    vertexFile.seekg(0, std::ios::beg); 
                }

                // allocate memory
                buffer = (char *)malloc(fileSize * sizeof(char));
                vertexFile.read(buffer, fileSize); // read data as one big block
                size_t start = 0;
                std::string delimiter = "|";

                // read buffer and do the magic ...
                for (size_t i = 0; i < fileSize; ++i)
                {
                    if (buffer[i] == '\n')
                    {
                        // get a row into string form buffer with start- and end-point
                        std::string row(&buffer[start], &buffer[i]);

                        // remove unnecessary '\n' at the beginning of a string
                        if (row.find('\n') != std::string::npos)
                        {
                            row.erase(0, 1);
                        }

                        size_t last = 0;
                        size_t next = 0;

                        // first line of *.csv contains the attributes -> write to attributes vector
                        if (start == 0)
                        {
                            std::string property_key;
                            Ldbc_Data_Type data_type;
                            // extract attribute from delimiter, e.g. id|name|url to id,name,url and push back to attributes vector
                            while ((next = row.find(delimiter, last)) != std::string::npos)
                            {
                                property_key = row.substr(last, next - last);
                                data_type = get_data_type(vertexType, property_key);
                                if (data_type == Ldbc_Data_Type::ERROR) {
                                    throw std::invalid_argument(file.string() + ":" + vertexType + ":" + property_key + " could not be found in schema");
                                }
                                attributes.push_back(std::make_pair(property_key, data_type));
                                last = next + 1;
                            }
                            // last attribute
                            property_key = row.substr(last);
                            data_type = get_data_type(vertexType, property_key);
                            if (data_type == Ldbc_Data_Type::ERROR) {
                                throw std::invalid_argument(file.string() + ":" + vertexType + ":" + property_key + " could not be found in schema");
                            }
                            attributes.push_back(std::make_pair(property_key, data_type));
                        }
                        else
                        {
                            // actual data:
                            std::unordered_map<std::string, property_type> properties;
                            size_t attrIndex = 0;
                            std::string ldbcID = row.substr(0, row.find(delimiter));
                            while ((next = row.find(delimiter, last)) != std::string::npos)
                            {   
                                auto key_to_datatype = attributes[attrIndex];
                                property_type property_value = convert_property_value(row.substr(last, next - last), key_to_datatype.second);
                                properties.insert(std::make_pair(key_to_datatype.first, property_value));
                                last = next + 1;
                                ++attrIndex;
                            }
                            // last attribute
                            auto key_to_datatype = attributes[attrIndex];
                            property_type propertyValue = convert_property_value(row.substr(last), key_to_datatype.second);
                            properties.insert(std::make_pair(key_to_datatype.first, propertyValue));

                            //-----------------------------------------------------
                            // create vertex and insert into graph with properties
                            uint64_t systemID = graph.add_vertex(vertexTypeNumber, properties);

                            // map vertexType and ldbc id to system generated id
                            globalIdLookupMap.insert({{vertexType, ldbcID}, systemID});
                            //-----------------------------------------------------
                            properties.clear(); // free memory
                        }

                        start = i; // set new starting point for buffer (otherwise it's concatenated)
                    }
                }

                free(buffer); // free memory
                vertexFile.close();

                ++vertexTypeNumber;
                attributes.clear();
            }
        }

        // function which returns the vertex_type_number (has no value if non existing)
        std::optional<int> get_vertex_type_number(const std::string &vertexType) {
            // iterate through entities-map to look up for paramater
            for (auto const &entry : vertexTypeLookup) {
                if (entry.second == vertexType) {
                    return entry.first;
                }
            }

            return {};
        }

        // function which returns true, if the edge type already exist
        bool exist_edge_type_name(const std::string &edge_type) {
            // iterate through edges-map to look up for paramater
            for (auto const &entry : edgeTypeLookup) {
                if (entry.second == edge_type) {
                    return true;
                }
            }

            return false;
        }

        // for debugging
        void print_file_names() {
            std::cout << "File-directory: " << getDirectory() << std::endl;
            std::cout << "Vertices-Files: " << std::endl;
            for (const auto &v : verticesPaths) {
                std::cout << "\t" << v << std::endl;
            }
            std::cout << "Edge-Files: " << std::endl;
            for (const auto &rel : edgesPaths) {
                std::cout << "\t" << rel << std::endl;
            }

        }

        // function which clears all intermediates after import
        void clear_intermediates() {
            std::cout << "CleanUp";
            globalIdLookupMap.clear();
            edgeTypeLookup.clear();
            vertexTypeLookup.clear();
            edgesPaths.clear();
            edgeProperties.clear();
            verticesPaths.clear();
            vertexEdgesLookup.clear();
        }

        // function which returns the total number of edges (IMPORTANT: vertex generation has to be done first, because of the vertexType lookup creation)
        uint64_t get_total_number_edges() {

            uint64_t result = 0;

            if (!edgesPaths.empty()) {

                // iterate through vector of edge-type file paths
                for (const auto &file : edgesPaths) {
                    std::string edge_type = getEntityType(file);


                    // TOdo: use regExp ([a-zA-Z]+)_([a-zA-Z]+)_([a-zA-Z]+)
                    std::string sourceVertexType = edge_type.substr(0, edge_type.find('_'));
                    edge_type.erase(0, edge_type.find('_') + 1);

                    std::string edgeType = edge_type.substr(0, edge_type.find('_'));
                    edge_type.erase(0, edge_type.find('_') + 1);

                    std::string targetVertexType = edge_type;

                    char *buffer;

                    uint64_t fileSize = 0;

                    std::ifstream edgeFile(file, std::ios::binary |
                                                        std::ios::ate); // 'ate' means: open and seek to end immediately after opening

                    if (!edgeFile) {
                        std::cerr << "Error, opening file. ";
                        exit(EXIT_FAILURE);
                    }

                    // calculate file size
                    if (edgeFile.is_open()) {
                        fileSize = static_cast<uint64_t>(edgeFile.tellg()); // tellg() returns: The current position of the get pointer in the stream on success, pos_type(-1) on failure.
                        edgeFile.clear();
                        edgeFile.seekg(0, std::ios::beg); // Seeks to the very beginning of the file, clearing any fail bits first (such as the end-of-file bit)
                    }

                    // allocate memory
                    buffer = (char *) malloc(fileSize * sizeof(char));
                    edgeFile.read(buffer, fileSize); // read data as one big block
                    bool firstLine = true;

                    // check from file name whether it's a edge file or multi value attribute file
                    if (get_vertex_type_number(targetVertexType).has_value()) {

                        for (size_t i = 0; i < fileSize; ++i) {
                            if (buffer[i] == '\n') {
                                // skip first line (attributes infos....)
                                if (firstLine) {
                                    firstLine = false;
                                } else {
                                    ++result;
                                }
                            }
                        }

                    }

                    free(buffer); // free memory
                    edgeFile.close();

                }
            }
            return result;
        }

        // get number of vertices from files and fill vertexTypeDictionary
        uint64_t get_total_number_vertices() {

            uint64_t result = 0;

            // iterate through vector of vertex-addresses
            for (const auto &file : verticesPaths) {
                char *buffer;
                uint64_t fileSize = 0;
                std::ifstream vertexFile(file, std::ios::binary |
                                                      std::ios::ate); // 'ate' means: open and seek to end immediately after opening

                if (!vertexFile) {
                    std::cerr << "Error, opening file. ";
                    exit(EXIT_FAILURE);
                }

                // calculate file size
                if (vertexFile.is_open()) {
                    fileSize = static_cast<uint64_t>(vertexFile.tellg()); // tellg() returns: The current position of the get pointer in the stream on success, pos_type(-1) on failure.
                    vertexFile.clear();
                    // Seeks to the very beginning of the file, clearing any fail bits first (such as the end-of-file bit)
                    vertexFile.seekg(0, std::ios::beg);
                }

                // allocate memory
                buffer = (char *)malloc(fileSize * sizeof(char));
                vertexFile.read(buffer, fileSize); // read data as one big block
                size_t start = 0;
                std::string delimiter = "|";

                // read buffer and do the magic ...
                for (size_t i = 0; i < fileSize; ++i) {
                    if (buffer[i] == '\n')
                    {
                        // get a row into string form buffer with start- and end-point
                        std::string row(&buffer[start], &buffer[i]);

                        // remove unnecessary '\n' at the beginning of a string
                        if (row.find('\n') != std::string::npos) {
                            row.erase(0, 1);
                        }

                        // first line of *.csv contains the attributes -> write to attributes vector
                        if (start != 0) {
                            ++result;
                        }

                        // set new starting point for buffer (otherwise it's concatenated)
                        start = i;
                    }
                }

                free(buffer); // free memory
                vertexFile.close();
            }
            return result;
        }

        // this function reads the edge-files and fills the intermediate: vertexEdgeLookup
        // + creates the edgeLookup (number to string) for the graph
        void fill_vertexEdgesLookup(Graph& graph){

            if(!edgesPaths.empty()) {
                std::cout << "(2/2) Generating LDBC-Edges ...";
                std::cout.flush();

                //this variable is used for the edgeLookup-keys, starting by 0
                unsigned short int edgeTypeNumber = 0;

                // iterate through vector of vertex-addresses
                for (const auto &file : edgesPaths) {
                    // get the edge-infos from file name: e.g. ([...path...] / [person_likes_comment].csv) --> person_likes_comment
                    // TODO: use regExp
                    std::string edge_type = getEntityType(file);
                    std::string sourceVertexType = edge_type.substr(0, edge_type.find('_'));
                    edge_type.erase(0, edge_type.find('_') + 1);

                    std::string edgeType = edge_type.substr(0, edge_type.find('_'));
                    edge_type.erase(0, edge_type.find('_') + 1);

                    std::string targetVertexType = edge_type;

                    char* buffer;

                    uint64_t fileSize = 0;

                    std::ifstream edgeFile(file, std::ios::binary | std::ios::ate); // 'ate' means: open and seek to end immediately after opening

                    if (!edgeFile) {
                        std::cerr << "Error, opening file. ";
                        exit(EXIT_FAILURE);
                    }

                    // calculate file size
                    if (edgeFile.is_open()) {
                        fileSize = static_cast<uint64_t>(edgeFile.tellg()); // tellg() returns: The current position of the get pointer in the stream on success, pos_type(-1) on failure.
                        edgeFile.clear();
                        edgeFile.seekg(0, std::ios::beg); // Seeks to the very beginning of the file, clearing any fail bits first (such as the end-of-file bit)
                    }

                    // allocate memory
                    buffer = (char*) malloc( fileSize * sizeof( char ) );
                    edgeFile.read(buffer, fileSize); // read data as one big block

                    size_t start = 0;
                    std::string delimiter = "|";

                    // check from file name whether it's an edge file or multi value attribute file
                    if(!get_vertex_type_number(targetVertexType).has_value()) {
                        // Multi-value-attributes: just take the last recently one
                        std::string propertyKey;
                        Ldbc_Data_Type data_type = Ldbc_Data_Type::STRING;
                        std::unordered_map<uint64_t, property_type> multiValueAttr;
                        uint64_t systemID;
                        property_type value;
			
                        for(size_t i = 0; i < fileSize; ++i){
                            if(buffer[i] == '\n'){
                                // get a row into string form buffer with start- and end-point
                                std::string row(&buffer[start], &buffer[i]);

                                // remove unnecessary '\n' at the beginning of a string
                                if(row.find('\n') != std::string::npos){
                                    row.erase(0,1);
                                }

                                // first line: get the attribute a.k.a key for the property, e.g. Person.id|email -> get 'email'
                                if(start == 0){
                                    propertyKey = row.substr(row.find(delimiter) + 1);
                                    data_type = get_data_type(sourceVertexType ,propertyKey);
                                    if (data_type == Ldbc_Data_Type::ERROR)
                                        throw std::invalid_argument(file.string() + ":" + edgeType + ":" + propertyKey + " could not be found in schema");
                                    
                                }else{
                                    // (1) write data to vector: if key is already present, over write value (simplicity: we take the newest one)
                                    systemID = globalIdLookupMap[{sourceVertexType, row.substr(0, row.find(delimiter))}];
                                    value = convert_property_value(row.substr(row.find(delimiter) + 1), data_type);
                                    multiValueAttr[systemID] = std::move(value);
                                }

                                start = i; // set new starting point for buffer (otherwise it's concatenated)
                            }
                        }
                        // iterate through multiValue map and assign property to vertex
                        for(const auto &pair : multiValueAttr){
                            //const std::pair<std::string, std::string> keyValuePair = {propertyKey, pair.second};
                            graph.add_property_to_vertex(pair.first, {propertyKey, pair.second});
                        }

                    }
                    // handling of edge-files ...
                    else{
                        // check if the name already exists
                        if (!exist_edge_type_name(edgeType)) {
                            ++edgeTypeNumber;
                            edgeTypeLookup.insert(std::make_pair(edgeTypeNumber, edgeType));
                        }
			
                        bool hasProperties = false;
                        std::string propertyKey;
                        uint64_t sourceVertexId, targetVertexId;

                        // read buffer and do the magic ...
                        for(size_t i = 0; i < fileSize; ++i){
                            if(buffer[i] == '\n'){
                                // get a row into string form buffer with start- and end-point
                                std::string row(&buffer[start], &buffer[i]);

                                // remove unnecessary '\n' at the beginning of a string
                                if(row.find('\n') != std::string::npos){
                                    row.erase(0,1);
                                }

                                size_t last = 0;
                                size_t next = 0;
                                size_t count = 0;

                                // first line of *.csv: Differentiate whether it's
                                // (1) edge without properties: e.g. Person.id|Person.id -> #delimiter = 1
                                // (2) edge with properties: e.g. Person.id|Person.id|fromDate -> #delimiter = 2
                                if(start == 0){
                                    // if there are 2 delimiter ('|') -> edge file with properties
                                    while ((next = row.find(delimiter, last)) != std::string::npos){
                                        last = next + 1;
                                        ++count;
                                    }
                                    if(count == 2){
                                        hasProperties = true;
                                        propertyKey = row.substr(last);
                                    }
                                } else {
                                    // lines of data: (from_local-ldbc-id), (to_local-ldbc-id) and property
                                    // get the system-(global) id's from local ids
                                    sourceVertexId = globalIdLookupMap.at({sourceVertexType, row.substr(0, row.find(delimiter))});
                                    // remove from id from string
                                    row.erase(0, row.find(delimiter) + delimiter.length());
                                    std::string value;
                                    if(!hasProperties){
                                        // WITHOUT properties: just from the first delimiter on
                                        targetVertexId = globalIdLookupMap.at({targetVertexType, row});

                                        // insert edge into vertexRealtionsLookup:
                                        vertexEdgesLookup[sourceVertexId].push_back(morphstore::Edge(sourceVertexId, targetVertexId, edgeTypeNumber));
                                    }else{
                                        // with properties means: toID is until the next delimiter, and then the value for the property
                                        targetVertexId = globalIdLookupMap.at({targetVertexType, row.substr(0, row.find(delimiter))});
                                        row.erase(0, row.find(delimiter) + delimiter.length());
                                        value = row;

                                        // insert edge into vertexEdgesLookup with its edge-property:
                                        auto edge = morphstore::Edge(sourceVertexId, targetVertexId, edgeTypeNumber);
                                        vertexEdgesLookup[sourceVertexId].push_back(edge);
                                        // assuming all properties of an edge are defined in the same file
                                        edgeProperties[edge.getId()] = {{propertyKey, value}};
                                    }
                                }
                                start = i; // set new starting point for buffer (otherwise it's concatenated)
                            }
                        }
                    }
                    free(buffer); // free memory
                    edgeFile.close();
                }
                // graph gets full edge-type-list here:
                graph.setEdgeTypeDictionary(edgeTypeLookup);
            }
        }

        // TODO: is this function really needed?
        // function for sorting the vertexEdgesLookup ASC (needed in CSR)
        // sorting for every vertex its vector list with target-ids ASC
        void sort_VertexEdgesLookup(){
            // sorting the first element of the pair (target-id)
            for(auto &rel: vertexEdgesLookup){
                std::sort(rel.second.begin(), rel.second.end());
            }
        }

        // this function writes the actual data from the intermediate vertexEdgesLookup into the graph
        void generate_edges(Graph&  graph){
            std::cout << " Writing edges into graph " << std::endl;
            // firstly, sorting the intermediates with their target IDs ASC
            sort_VertexEdgesLookup();

            uint64_t graphSize = graph.getVertexCount();

            for(uint64_t vertexID = 0; vertexID < graphSize ; ++vertexID){
                auto edges = vertexEdgesLookup[vertexID];
                // add edge data:
                graph.add_edges(vertexID, edges);
                for(auto edge: edges) {
                    auto entry = edgeProperties.find(edge.getId());
                    if (entry != edgeProperties.end()) {
                        graph.set_edge_properties(entry->first, entry->second);
                    }
                }
            }
        }

        void generate_vertex_type_lookup() {
            uint64_t vertex_type_number = 0;
            for(std::string vertex_file: verticesPaths) {
                vertexTypeLookup.insert(std::make_pair(vertex_type_number, getEntityType(vertex_file)));
                vertex_type_number++;
            }
        }

        // MAIN IMPORT FUNCTION: see steps in comments
        void import(Graph&  graph) {
            std::cout << "Importing LDBC-files into graph ... ";
            std::cout.flush();

            // (1) get number vertices and number edges:
            uint64_t numberVertices = get_total_number_vertices();

            // populate vertex_type_lookup for differentiating between edge and property files
            generate_vertex_type_lookup();
            graph.set_vertex_type_dictionary(vertexTypeLookup);

            uint64_t numberEdges = get_total_number_edges();

            // (2) allocate graph memory
            graph.allocate_graph_structure(numberVertices, numberEdges);

            // (3) generate vertices
            generate_vertices(graph);
	    
            // (4) read edges and write to intermediate results
            fill_vertexEdgesLookup(graph);

            // (5) read intermediates and write edges
            generate_edges(graph);

            // (6) clear intermediates
            clear_intermediates();

            std::cout << "--> done" << std::endl;
        }
    };
}

#endif //MORPHSTORE_LDBC_IMPORT_H
