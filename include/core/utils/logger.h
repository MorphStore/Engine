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
 * @file logger.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_UTILS_LOGGER_H
#define MORPHSTORE_CORE_UTILS_LOGGER_H

#include "cmake_template.h"

#include <core/memory/stl_wrapper/ostream.h>

#include <array>

#define MSV_LOG_LEVEL_COUNT 6
#ifndef MSV_GIT_BRANCH
#  define MSV_GIT_BRANCH
#endif
#ifndef MSV_GIT_HASH
#  define MSV_GIT_HASH
#endif

namespace morphstore {

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

         char const * get_name() const {
            return m_LogLevelName;
         }
         char const * get_color() const {
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
      const char * get_tag_text( int p_LogLevel ) const {
         return m_LevelsAndColors[ p_LogLevel ].m_LogLevelName;
      }
      const char * get_tag_color( int p_LogLevel ) const {
         return m_LevelsAndColors[ p_LogLevel ].m_ColorText;
      }
      const char * get_head( void ) const {
         return m_LineStartText;
      }
      char const * get_tail( ) const {
         return m_LineEndText;
      }
      const char * get_color_default( void ) const {
         return m_ColorDefaultText;
      }
      const char * get_entry_start( void ) const {
         return m_EntryBeginText;
      }
      const char * get_entry_end( void ) const {
         return m_EntryEndText;
      }


};
class html_formatter : public formatter {
   public:
      html_formatter( void ) :
         formatter{
            "<tr>",
            "</tr>\n",
            "<td>",
            "</td>\n",
            "",
            {
               levels_colors( "[Trace]:</td>\n", "<td bgcolor=\"#FFFFFF\">" ),
               levels_colors( "[Debug]:</td>\n", "<td bgcolor=\"#0000FF\">" ),
               levels_colors( "[Info ]:</td>\n", "<td bgcolor=\"#00FF00\">" ),
               levels_colors( "[Warn ]:</td>\n", "<td bgcolor=\"#FFCC66\">" ),
               levels_colors( "[Error]:</td>\n", "<td bgcolor=\"#FF0000\">" ),
               levels_colors( "[WTF  ]:</td>\n", "<td bgcolor=\"#FF00FF\">" )
            }
         }{ }
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
               levels_colors( "[Trace]: ", "\033[0m" ),
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
      virtual morphstore::ostream & get_out( void ) = 0;
   public:

      template< typename T, typename ... Args >
      void log( T p_LogLevel, const char * p_From, Args &&... p_Args ) {
         head( p_LogLevel );
         log_message_from_line( p_From );
         get_out( ) << get_formatter( ).get_entry_start( );
         log_message_line( p_Args... );
         get_out( ) << get_formatter( ).get_entry_end( );
         tail( );
      }

   private:
      template< typename ... Args >
      void log_message_line( Args &&... args ) {
         using isoHelper = int[];
         ( void ) isoHelper{
            0, ( void( get_out( ) //<< get_formatter( ).entry_start()
                           << std::forward< Args >( args )
                           /*<< get_formatter( ).entry_end( )*/ ), 0 ) ...
         };
      }

      void log_message_from_line( const char * p_From ) {
         get_out( ) << get_formatter( ).get_entry_start( ) << p_From << get_formatter( ).get_entry_end( );
      }


      void head( int p_LogLevel ) {
         morphstore::ostream & ttt = get_out( );
         ttt << get_formatter( ).get_head()
                     << get_formatter( ).get_tag_color( p_LogLevel )
                     << get_formatter( ).get_tag_text( p_LogLevel )
                     << get_formatter( ).get_color_default( );
      }

      void tail( void ) {
         get_out( ) << get_formatter( ).get_tail( );
      }
};

class shell_logger : public logger {
   private:
      shell_formatter m_Formatter;
      morphstore::ostream stream;
      shell_logger( void ):
         logger{} {
         log_header();
      }
   protected:
      void log_header( void ) override {
         get_out()   << "Project [ProjectName] started.\n"
                     << "Build Specs: \n"
                     << "Branch: " << MSV_GIT_BRANCH << "\n"
                     << "Commit: " << MSV_GIT_HASH << "\n";
      }
      void log_footer( void ) override {}

      morphstore::ostream & get_out( void ) override {
         return stream;
      }

      inline formatter const & get_formatter( void ) const override {
         return m_Formatter;
      }
   public:
      static shell_logger & get_instance(void) {
         static shell_logger instance;
         return instance;
      }

};

class html_logger : public logger {
   private:
      html_formatter m_Formatter;
      morphstore::ostream stream;
   protected:
      void log_header( void ) override {
         get_out( )
            << "<!DOCTYPE html><html><head><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            << "<style>"
            << "table {"
            << " border-collapse: collapse;border-spacing: 0;width: 100%;border: 1px solid #ddd;"
            << "}"
            << "th, td {"
            << "  text-align: center;"
            << "}"
            << "tr:nth-child(even) {"
            << " background-color: #f2f2f2"
            << "}"
            << "[data-toggle=\"toggle\"] {"
            << " display: none;"
            << "}"
            << "</style>"
            << "</head><body>"
            << "Project [ProjectName] started.\n"
            << "Build Specs: \n"
            << "Branch: " << MSV_GIT_BRANCH << "\n"
            << "Commit: " << MSV_GIT_HASH << "\n"
            << "<table>\n";
      }
      void log_footer( void ) override {
         get_out( )
            << "</table></bodý></html>";
      }
      inline morphstore::ostream & get_out( void ) override {
         return stream;
      }

      inline formatter const & get_formatter( void ) const override {
         return m_Formatter;
      }
   public:

      static html_logger & get_instance( void ) {
         static thread_local html_logger html_logger_instance;
         return html_logger_instance;
      }
      ~html_logger( ) {
         log_footer( );
      }
};






#ifdef MSV_NO_SHELL_LOGGER
//html logger is assumed
typedef shell_logger morphstore_logger;
#else
//html_logger log_instance;
typedef shell_logger morphstore_logger;
#endif

}


#ifdef MSV_NO_LOG
#  define trace(...)
#  define debug(...)
#  define info(...)
#  define warn(...)
#  define error(...)
#  define wtf(...)
#else
#  ifdef DEBUG
#     define trace(...) morphstore::morphstore_logger::get_instance( ).log( 0, __FUNCTION__, __VA_ARGS__ )
#     define debug(...) morphstore::morphstore_logger::get_instance( ).log( 1, __FUNCTION__, __VA_ARGS__ )
#     define info(...) morphstore::morphstore_logger::get_instance( ).log( 2, __FUNCTION__, __VA_ARGS__ )
#  else
#     define trace(...)
#     define debug(...)
#     define info(...)
#  endif
#  define warn(...) morphstore::morphstore_logger::get_instance( ).log( 3, __FUNCTION__, __VA_ARGS__ )
#  define error(...) morphstore::morphstore_logger::get_instance( ).log( 4, __FUNCTION__, __VA_ARGS__ )
#  define wtf(...) morphstore::morphstore_logger::get_instance( ).log( 5, __FUNCTION__, __VA_ARGS__ )
#endif




#endif //MORPHSTORE_CORE_UTILS_LOGGER_H
