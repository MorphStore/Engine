@page TextFormatting Text Formatting

@tableofcontents

This page provides a set of examples how you can format text in doxygen.
For more detailed information about specific commands check the doxygen documentation.

A good start is the [Markdown support page](https://www.stack.nl/~dimitri/doxygen/manual/markdown.html).


---

@section structural_commands Structural commands

It is possible to sturcture your page into sections, subsections and subsubsections.
To list them in a table of contents (for example look at the top right of this page) you need to call \@tableofcontents at the top of your page.

####Input

@code

    @page yourPageHandle Your page title

    @tableofcontents

    @section 1. This is a section (level 1)
    @subsection 1.1 This is a subsection of section (level 2)
    @subsubsection 1.1.1 This is a subsubsection of subsection (level 3)

@endcode

####Output

# 1. This is a section (level 1)

##1.1 This is a subsection of section (level 2)

###1.1.1 This is a subsubsection of subsection (level 3)

---

@section horizontal_line Horizontal line
To create a horizontal line like above and beneath this section:

####Input

@code
    
    ---

@endcode

####Output

---
---


@section Formatting

@subsection italic Italic, bold

####Input

@code

    _italic text_

    __bold text__

@endcode

####Output

_italic text_

__bold text__


---

@section Links

In Doxygen you can link almost everything with one command: @@ref

Pages, Classes, Files, Datastructures ...

Note: Links to pages need the page handle (first argument of @@page) and is replaced by the title of the page

Links to websites are automatically detected and don't need the @@ref command, but you can alter the caption.

####Input

@code

    Page:                         @ref JoinTest \n
    Class:                        @ref eris::JoinTest \n
    Function:                     @ref eris::JoinTest::run \n
    File:                         @ref JoinTest.cpp \n
    Datatype:                     @ref eris::EPtr \n
    Website:                      https://eris-platform.de/ \n
    __Altered Captions__ \n
    Page:                         [HowTo do a join in ERIS](@ref JoinTest) \n
    Class:                        [The JoinTest class](@ref eris::JoinTest) \n
    Website:                      [ERIS Homepage](https://eris-platform.de/) \n

@endcode

####Output

Page:                         @ref JoinTest \n
Class:                        @ref eris::JoinTest \n
Function:                     @ref eris::JoinTest::run \n
File:                         @ref JoinTest.cpp \n
Datatype:                     @ref eris::EPtr \n
Website:                      https://eris-platform.de/ \n
__Altered Captions__ \n
Page:                         [HowTo do a join in ERIS](@ref JoinTest) \n
Class:                        [The JoinTest class](@ref eris::JoinTest) \n
Website:                      [ERIS Homepage](https://eris-platform.de/) \n


---


@subsection Colors

Doxygen does not support commands to colorize text by default. But there are already some custom commands to fix that.

If you want to know how to add custom commands, see https://www.stack.nl/~dimitri/doxygen/manual/custcmd.html

####Input

@code

    @red{Normal red text} \n
    @lred{Light red text} \n
    @dred{Dark red text} \n
    @green{Normal green text} \n
    @lgreen{Light green text} \n
    @dgreen{Dark green text} \n

@endcode

####Output
 
@red{Normal red text} \n
@lred{Light red text} \n
@dred{Dark red text} \n
@green{Normal green text} \n
@lgreen{Light green text} \n
@dgreen{Dark green text} \n

---

@subsection Fontsize

To change the font size of text (the argument is an integer interpreted as pixels)

####Input

@code

    This is normal sized text.
    @fontsize{24} This is bigger text. @endfontsize
    This is normal sized text, again.

@endcode

####Output

This is normal sized text.
@fontsize{24} This is bigger text. @endfontsize
This is normal sized text, again.

---

@subsection custom_links Custom links

The @@expl command is used to highlight references to manually written pages.

####Input

@code

    @expl{JoinTest}

@endcode

####Output

@expl{JoinTest}

---

@section codeBlocks Code Blocks

Standard code blocks

####Input

@box{
@line{@@code}
@line{&nbsp;&nbsp;&nbsp;&nbsp;void foo()\{}
@line{&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;std::cout << "bar" << std::endl;}
@line{&nbsp;&nbsp;&nbsp;&nbsp;\}}
@line{@@endcode}
}

####Output

@code
    void foo(){
        std::cout << "bar" << std::endl;
    }
@endcode

Doxygen also supports language highlightning for some languages.

####Input

@code
    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}

        void foo(){
            std::cout << "bar" << std::endl;
        }

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@endcode

####Output

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}

    void foo(){
        std::cout << "bar" << std::endl;
    }

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code has to be encapsulated by 2 lines with at least 3 tilde (~) and an optional language definition after the opening line.


---

@section html_support HTML Support

Most of HTML tags are passed through doxygen, so you can format text like in HTML using CSS.

####Input

@code
    
    <span style="color: #ff0000; border: 1px solid #68707c; padding: 2px">This is a red text with bluish border</span>

@endcode

####Output

<span style="color: #ff0000; border: 1px solid #68707c; padding: 2px">This is a red text with bluish border</span>

---

@section Headings

####Input

@code

    # Heading Level 1 

    ## Heading Level 2

    ### Heading Level 3

    #### Heading Level 4

    ##### Heading Level 5

    ###### Heading Level 6

@endcode

####Output

# Heading Level 1 

## Heading Level 2

### Heading Level 3

#### Heading Level 4

##### Heading Level 5

###### Heading Level 6



---

@author Eric Mier \<eric.mier@tu-dresden.de\>
