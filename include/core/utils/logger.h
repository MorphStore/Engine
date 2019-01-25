/**********************************************************************************************
 * Copyright (C) 2019 by Johannes Pietrzyk                                                    *
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
 * @file logger.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_UTILS_LOGGER_H
#define MORPHSTORE_CORE_UTILS_LOGGER_H

#include "cmake_template.h"

#include <string>
#include <array>
#include <iostream>


#define MSV_LOG_LEVEL_COUNT 5
#ifndef MSV_GIT_BRANCH
#  define MSV_GIT_BRANCH
#endif
#ifndef MSV_GIT_HASH
#  define MSV_GIT_HASH
#endif

namespace morphstore { namespace logging {

class formatter {
   public:
      struct levels_colors {
         char const * m_LogLevelName;
         char const * m_ColorText;
         levels_colors( void ) : m_LogLevelName{""}, m_ColorText{""}{ }
         levels_colors( char const * p_LogLevelName, char const * p_ColorText ):
            m_LogLevelName{ p_LogLevelName },
            m_ColorText{ p_ColorText }{ }
         levels_colors( levels_colors const & ) = default;
         levels_colors( levels_colors && ) = default;
         levels_colors & operator=( levels_colors const & ) = default;
         levels_colors & operator=( levels_colors && ) = default;

         char const * name() const {
            return m_LogLevelName;
         }
         char const * color() const {
            return m_ColorText;
         }
      };
   protected:
      char const *  m_LineStartText;
      char const * m_LineEndText;
      char const * m_EntryBeginText;
      char const * m_EntryEndText;
      char const * m_ColorDefaultText;

      std::array< levels_colors, MSV_LOG_LEVEL_COUNT > m_LevelsAndColors;
   public:
      formatter(
         char const * p_LineStartText,
         char const * p_LineEndText,
         char const * p_EntryBeginText,
         char const * p_EntryEndText,
         char const * p_ColorDefaultText,
         std::array< levels_colors, MSV_LOG_LEVEL_COUNT >&& p_LevelsAndColors ) :
         m_LineStartText{ p_LineStartText },
         m_LineEndText{ p_LineEndText },
         m_EntryBeginText{ p_EntryBeginText },
         m_EntryEndText{ p_EntryEndText },
         m_ColorDefaultText{ p_ColorDefaultText },
         m_LevelsAndColors{ p_LevelsAndColors }{ }
      const char * tag_text( int p_LogLevel ) const {
         return m_LevelsAndColors[ p_LogLevel ].m_LogLevelName;
      }
      const char * tag_color( int p_LogLevel ) const {
         return m_LevelsAndColors[ p_LogLevel ].m_ColorText;
      }
      const char * head( void ) const {
         return m_LineStartText;
      }
      char const * tail( ) const {
         return m_LineEndText;
      }
      const char * color_default( void ) const {
         return m_ColorDefaultText;
      }
      const char * entry_start( void ) const {
         return m_EntryBeginText;
      }
      const char * entry_end( void ) const {
         return m_EntryEndText;
      }


};
class shell_formatter : public formatter {
   public:
      shell_formatter( void ) :
         formatter{
            "",
            "\n",
            "",
            " ",
            "\033[0m",
            {
               levels_colors( "[Debug]: ", "\033[1;34m" ),
               levels_colors( "[Info ]: ", "\033[1;32m" ),
               levels_colors( "[Warn ]: ", "\033[1;33m" ),
               levels_colors( "[Error]: ", "\033[1;31m" ),
               levels_colors( "[WTF  ]: ", "\033[1;35m" )
            }
         }{ }
};

class logger {
   protected:
      virtual void log_header( void ) = 0;
      virtual void log_footer( void ) = 0;
      virtual formatter const & get_formatter( void ) const = 0;
      virtual std::ostream & get_out( void ) = 0;
   public:

      template< typename T, typename ... Args >
      void log( T p_LogLevel, Args &&... p_Args ) {
         head( p_LogLevel );
         log_message_line( p_Args... );
         tail( );
      }

   private:
      template< typename ... Args >
      void log_message_line( Args &&... args ) {
         using isoHelper = int[];
         ( void ) isoHelper{
            0, ( void( get_out( ) << get_formatter( ).entry_start()
                           << std::forward< Args >( args )
                           << get_formatter( ).entry_end( )), 0 ) ...
         };
      }

      void head( int p_LogLevel ) {
         get_out( )  << get_formatter( ).head()
                     << get_formatter( ).tag_color( p_LogLevel )
                     << get_formatter( ).tag_text( p_LogLevel )
                     << get_formatter( ).color_default( );
      }

      void tail( void ) {
         get_out( ) << get_formatter( ).tail( );
      }
};

class shell_logger : public logger {
   private:
      shell_formatter m_Formatter;
   protected:
      void log_header( void ) override {
         get_out()   << "Project [ProjectName] started.\n"
                     << "Build Specs: \n"
                     << "Branch: " << MSV_GIT_BRANCH << "\n"
                     << "Commit: " << MSV_GIT_HASH << "\n";
      }
      void log_footer( void ) override {}
      std::ostream & get_out( void ) override {
         return std::cout;
      }

      inline formatter const & get_formatter( void ) const override {
         return m_Formatter;
      }
   public:
      shell_logger( void )  {
         log_header();
      }
};






#ifdef MSV_NO_SHELL_LOGGER
//...
#else
shell_logger log_instance;
#endif

}}


#ifdef MSV_NO_LOG
#  define debug(...)
#  define info(...)
#  define warn(...)
#  define error(...)
#  define wtf(...)
#else
#  ifdef DEBUG
#     define debug(...) morphstore::logging::log_instance.log( 0, __VA_ARGS__ )
#     define info(...) morphstore::logging::log_instance.log( 1, __VA_ARGS__ )
#  elif defined( MSV_DEBUG_MALLOC )
#     define debug(...) morphstore::logging::log_instance.log( 0, __VA_ARGS__ )
#  else
#     define debug(...)
#     define info(...)
#endif
#  define warn(...) morphstore::logging::log_instance.log( 2, __VA_ARGS__ )
#  define error(...) morphstore::logging::log_instance.log( 3, __VA_ARGS__ )
#  define wtf(...) morphstore::logging::log_instance.log( 4, __VA_ARGS__ )
#endif




#endif //MORPHSTORE_CORE_UTILS_LOGGER_H
