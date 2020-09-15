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


#ifndef MORPHSTORE_STRING_MANIPULATION_H
#define MORPHSTORE_STRING_MANIPULATION_H

#include <string>
#include <vector>


namespace morphstore {

	template<typename T>
	MSV_CXX_ATTRIBUTE_PPUNUSED
	inline
	std::string strify(T input){
		std::stringstream s;
		s << input;
		return s.str();
	}
	
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string str_lfill(std::string& str, uint64_t length, const char fillChar[1]){
		if(str.length() > length)
			return str;
		uint64_t fillup = length - str.length();
		std::stringstream s;
		for (unsigned i = 0; i < fillup; ++i) {
			s << fillChar;
		}
		s << str;
		return s.str();
	}
	
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string str_lfill(std::string&& str, uint64_t length, const char fillChar[1]){
		return str_lfill(str, length, fillChar);
	}
	
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string str_lfill(std::string & str, uint64_t length) {
		return str_lfill(str, length, " ");
	}
	
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string str_lfill(std::string && str, uint64_t length) {
		return str_lfill(str, length, " ");
	}
	
	
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string str_rfill(std::string& str, uint64_t length, const char fillChar[1]){
		if(str.length() > length)
			return str;
		uint64_t fillup = length - str.length();
		std::stringstream s;
		s << str;
		for (unsigned i = 0; i < fillup; ++i) {
			s << fillChar;
		}
		return s.str();
	}
	
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string str_rfill(std::string&& str, uint64_t length, const char fillChar[1]) {
		return str_rfill(str, length, fillChar);
	}
	
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string str_rfill(std::string& str, uint64_t length) {
		return str_rfill(str, length, " ");
	}
	
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string str_rfill(std::string&& str, uint64_t length) {
		return str_rfill(str, length, " ");
	}
	
	
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string str_mfill(std::string& str, uint64_t length, const char fillChar[1]){
		if(str.length() > length)
			return str;
		uint64_t fillup = (length - str.length()) / 2;
		std::string s = str_lfill(str, length - fillup, fillChar);
		return str_rfill(s, length, fillChar);
	}
	
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string str_mfill(std::string&& str, uint64_t length, const char fillChar[1]) {
		return str_mfill(str, length, fillChar);
	}
	
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string str_mfill(std::string& str, uint64_t length){
		return str_mfill(str, length, " ");
	}
	
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string str_mfill(std::string&& str, uint64_t length){
		return str_mfill(str, length, " ");
	}
	
	/**
	 * @brief Splits a string by given delimiter.
	 * @param input string to split
	 * @param delimiter
	 * @return vector of substrings
	 */
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::vector<std::string>* splitString(std::string input, const std::string& delimiter){
		std::vector<std::string>* split = new std::vector<std::string>();
		size_t pos = input.find(delimiter);
		while(pos != std::string::npos){
			split->push_back(input.substr(0, pos));
			input.erase(0, pos + delimiter.length());
			pos = input.find(delimiter);
		}
		split->push_back(input);
		return split;
	}
	
	/**
	 * @brief
	 * @param input
	 * @return
	 */
	MSV_CXX_ATTRIBUTE_PPUNUSED
	static
	std::string dotNumber(const std::string & input){
		std::string out;
		int64_t pos = input.length() - 1;
		uint8_t interval = 0;
		for(;pos >= 0; --pos){
			out = input.substr(pos,1) + out;
			++interval;
			if(interval == 3 && pos != 0){
				interval = 0;
				out = "," + out;
			}
		}
		return out;
	}
	
	
	
	/**
	 * @brief Trim from start (in place).
	 * @param s
	 */
	MSV_CXX_ATTRIBUTE_PPUNUSED
	inline
	void ltrim(std::string &s) {
		s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
			return !std::isspace(ch);
		}));
	}
	
	/**
	 * @brief Trim from end (in place).
	 * @param s
	 */
	MSV_CXX_ATTRIBUTE_PPUNUSED
	inline
	void rtrim(std::string &s) {
		s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
			return !std::isspace(ch);
		}).base(), s.end());
	}
	
	/**
	 * @brief Trim from both ends (in place).
	 * @param s
	 */
	MSV_CXX_ATTRIBUTE_PPUNUSED
	inline
	void trim(std::string &s) {
		ltrim(s);
		rtrim(s);
	}
} // morphstore

#endif //MORPHSTORE_STRING_MANIPULATION_H
