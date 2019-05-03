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
 * @brief vertex class and its functions
 * @todo Add data structure for properties
*/

#ifndef MORPHSTORE_VERTEX_H
#define MORPHSTORE_VERTEX_H

#include <core/storage/edge.h>

#include <vector>
#include <iostream>


namespace graph{

    class Vertex{

    private:
        unsigned long int id;
        unsigned long int ldbc_id;
        int entity;
        std::vector<Edge> adjList;

    public:

        Vertex(unsigned long int id, unsigned long int ldbc_id, int entity){
            SetVertex(id, ldbc_id, entity);
        }

        void SetVertex(unsigned long int id, unsigned long int ldbc_id, int entity){
            this->id = id;
            this->ldbc_id = ldbc_id;
            this->entity = entity;
        }

        unsigned long int getId() const{
            return id;
        }

        unsigned long int getLDBC_Id(){
            return ldbc_id;
        }

        int getEntity(){
            return entity;
        }

        const std::vector<Edge>& getAdjList() const{
            return adjList;
        }


        void setEntity(int newEntity){
            entity = newEntity;
        }

        bool deleteAdjList(){
            adjList.clear();
            if(adjList.size() == 0) return true;
            return false;
        }

        void addEdge(Edge e){
            this->adjList.push_back(e);
        }
    };
}

#endif //MORPHSTORE_VERTEX_H
