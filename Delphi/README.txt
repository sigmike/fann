FANN Delphi Binding
===================

These files are distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Author: Mauricio Pereira Maia <mauricio at uaisol.com.br>
Version: 1.1


Version History
===============

Version 1.0 - 10 Oct 2004 - First Version released.
Version 1.1 - 12 Nov 2004 - Fixed LoadFromFile bug.


Requeriments
============

- FANN Library Version 1.2.0
- Delphi 6 or 7. Prior Delphi versions should work too, but without the functions 
fann_create and fann_create_shorcut because of the variable argument list.)


HOW TO USE IT
=============

- Download the fann_win32_dll-1.2.0.zip file
- Put the file fannfloat.dll in your PATH. (If you want to use the fixed version
you should define FIXEDFANN on fann.pas)
- include fann.pas in your project and in your unit uses clause, and have fun!
- See the XorConsole sample for more details.


INSTALLING THE TFannNetwork
===========================

TFannNetwork is a Delphi component that encapsulates the Fann Library.
You do not have to install TFannNetwork to use Fann on Delphi, 
but it will make the library more Delphi friendly.
Currently it has only a small subset of all the library functions, but I 
hope that will change in the near future.

To install TFannNetwork you should follow all the previous steps and
- Copy the FannNetwork.pas and Fann.dcr to your Delphi Library PATH.
- Choose Component/Install Component.
- In the Unit file name field, click on Browse and point to the fannnetwork.pas file.
By default Delphi will install in the Borland User Components package, 
it might be changed using Package file name field or Into new package page.
- Click on Ok 
- A confirmation dialog will be shown asking if you want to build the package. Click on Yes.
- You have just installed TFannNetwork, now close the package window 
(Don't forget to put Yes when it ask if you want to save the package). 
- See the FannNetwork.pas file or the Xor Sample.

KNOWN PROBLEMS
==============

If you are getting in trouble to use your own compiled FANN DLL with Delphi that might be
because of the C++ naming mangle that changes between C++ compilers. 
You will need to make a TDUMP on your dll and changes all the name directives on fann.pas to 
the correct function names on your dll.


























