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
 * @file ldbc_schema.h
 * @brief Schema of the LDBC graph based on
 * https://raw.githubusercontent.com/ldbc/ldbc_snb_docs/dev/figures/schema-comfortable.png
 * @todo search for an existing Graph-Schema language (graph schemas should be stored in the resource folder)
 */

#ifndef MORPHSTORE_LDBC_SCHEMA_H
#define MORPHSTORE_LDBC_SCHEMA_H

#include <core/storage/graph/property_type.h>

#include <iostream>
#include <map>
#include <regex>

namespace morphstore {

    enum class Ldbc_Data_Type { LONG_STRING, STRING, TEXT, INT_32, ID, DATE_TIME, DATE, ERROR };

    // static not included -> f.i. hasTag edge seen as property tag.id
    static const std::map<std::string, std::map<std::string, Ldbc_Data_Type>> ldbc_schema{{
        // vertices
        {"person",
         {{"creationDate", Ldbc_Data_Type::DATE_TIME},
          {"firstName", Ldbc_Data_Type::STRING},
          {"lastName", Ldbc_Data_Type::STRING},
          {"gender", Ldbc_Data_Type::STRING},
          {"birthday", Ldbc_Data_Type::DATE},
          // !TODO actually an array of emails
          {"email", Ldbc_Data_Type::LONG_STRING},
          // !TODO actually an array of languages

          // (and not currently filled as csv header contains "language")
          //{"speaks", Ldbc_Data_Type::STRING},
          // TODO actually values for "speaks" array
          {"language", Ldbc_Data_Type::STRING},
          {"browserUsed", Ldbc_Data_Type::STRING},
          {"locationIP", Ldbc_Data_Type::STRING}}},
        {"forum", {{"creationDate", Ldbc_Data_Type::DATE_TIME}, {"title", Ldbc_Data_Type::LONG_STRING}}},
        {"post",
         {{"creationDate", Ldbc_Data_Type::DATE_TIME},
          {"browserUsed", Ldbc_Data_Type::STRING},
          {"locationIP", Ldbc_Data_Type::STRING},
          {"length", Ldbc_Data_Type::INT_32},
          // TODO: extra nullable type for the following 3: like TEXT?
          {"content", Ldbc_Data_Type::TEXT},
          {"language", Ldbc_Data_Type::STRING},
          {"imageFile", Ldbc_Data_Type::STRING}}},
        {"comment",
         {{"creationDate", Ldbc_Data_Type::DATE_TIME},
          {"browserUsed", Ldbc_Data_Type::STRING},
          {"locationIP", Ldbc_Data_Type::STRING},
          {"content", Ldbc_Data_Type::TEXT},
          {"length", Ldbc_Data_Type::INT_32}}},
        {"tagclass", {{"name", Ldbc_Data_Type::LONG_STRING}, {"url", Ldbc_Data_Type::LONG_STRING}}},
        {"tag", {{"name", Ldbc_Data_Type::LONG_STRING}, {"url", Ldbc_Data_Type::LONG_STRING}}},
        {"place",
         {{"name", Ldbc_Data_Type::LONG_STRING},
          {"url", Ldbc_Data_Type::LONG_STRING},
          {"type", Ldbc_Data_Type::STRING}}},
        {"organisation",
         {{"name", Ldbc_Data_Type::LONG_STRING},
          {"type", Ldbc_Data_Type::STRING},
          {"url", Ldbc_Data_Type::LONG_STRING}}},
        // edges
        {"likes", {{"creationDate", Ldbc_Data_Type::DATE_TIME}}},
        {"hasMember", {{"joinDate", Ldbc_Data_Type::DATE_TIME}}},
        {"hasModerator", {}},
        {"hasCreator", {}},
        {"hasTag", {}},
        {"containerOf", {}},
        {"replyOf", {}},
        {"isSubclassOf", {}},
        {"isPartOf", {}},
        {"isLocatedIn", {}},
        {"studyAt", {{"classYear", Ldbc_Data_Type::INT_32}}},
        {"workAt", {{"workFrom", Ldbc_Data_Type::INT_32}}},
        {"knows", {{"creationDate", Ldbc_Data_Type::DATE_TIME}}},
    }};

    Ldbc_Data_Type get_data_type(std::string entity_type, std::string property_key) {
        auto perEntity = ldbc_schema.find(entity_type);
        if (perEntity != ldbc_schema.end()) {
            auto propertiesMap = perEntity->second;
            auto propertyEntry = propertiesMap.find(property_key);
            if (propertyEntry != propertiesMap.end()) {
                return propertyEntry->second;
            }
        }

        // ldbc id is saved as an extra property as morphstore::graph generates new ones
        // static part of social network not included thus saved as property (!!wrongly!!)
        if (property_key == "id")
            return Ldbc_Data_Type::ID;

        // std::cout << "Could not find a data type for " << entity_type << " " << property_key;
        return Ldbc_Data_Type::ERROR;
    }

    property_type convert_property_value(std::string value, Ldbc_Data_Type type) {
        property_type converted_value;

        switch (type) {
        case Ldbc_Data_Type::INT_32:
            converted_value = std::stoi(value);
            break;
        case Ldbc_Data_Type::ID:
            converted_value = std::stoull(value);
            break;
        default:
            converted_value = value;
        };

        return converted_value;
    }
} // namespace morphstore

#endif // MORPHSTORE_PROPERTY_TYPE_H