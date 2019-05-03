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
 * @file vertex.h
 * @brief edge class and its functions
 * @todo Add data structure for property
*/

#ifndef MORPHSTORE_EDGE_H
#define MORPHSTORE_EDGE_H

#include <core/storage/vertex.h>

namespace graph{

    class Vertex;

    class Edge{

    private:
        Vertex* target;
        int relation;

    public:
        Edge(Vertex* target, int relation){
            target = target;
            relation = relation;
        }

        Vertex* getTarget() const{
            return target;
        }

        int getRelation() const{
            return relation;
        }

        void setTarget(Vertex* targetVertex){
            this->target = targetVertex;
        }

        void setRelation(int rel){
            this->relation = rel;
        }
    };
}



#endif //MORPHSTORE_EDGE_H
