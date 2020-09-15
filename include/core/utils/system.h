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


#ifndef MORPHSTORE_SYSTEM_H
#define MORPHSTORE_SYSTEM_H

#include <execinfo.h>
#include <string>
#include <sys/stat.h>
#include "string_manipulation.h"

namespace morphstore {

	/**
	 * @brief Checks if a given directory exists and creates it, if not.
	 * @param path to create
	 * @return
	 */
	static bool testAndCreateDirectory(const std::string path) {
		// Check if output directory exists, if not, create it
		struct stat buffer;
		
		auto pathParts = splitString(path, "/");
		std::string currentPath = "";
		for(auto & pp : *pathParts){
			currentPath += pp;
			if ((stat (currentPath.c_str(), &buffer) != 0) && (mkdir(currentPath.c_str(), 0777) == -1)) {
				std::cerr << "Could not create directory \"" << currentPath << "\"\nError: " << strerror(errno) << std::endl;
				return false;
			}
			currentPath += "/";
		}
		return true;
	}
}


#endif //MORPHSTORE_SYSTEM_H
