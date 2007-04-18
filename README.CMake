CMake support is added to FANN as an alternative to Autotools.

To compile fann using CMake you need the cmake command from www.cmake.org.

in order to build FANN follw theese steps:

1) Install CMake 
	$ sudo apt-get install cmake #if you are using ubuntu 
	# yum install cmake # on redhat

2) Build FANN
	$ cd fann-x.y.x
	$ mkdir build-cmake # (optional) out of sources build
	$ cmake .. # use ccmake if you prefer a ncurses like gui
	$ sudo make install # or make followed by sudo make install

If you want to install in the default install path (/usr/local on linux) 
just follow the above instructions.

If you want to install to another path just change the install prefix path using
-DCMAKE_INSTALL_PREFIX=/another/path/of/your/choice in the cmake (or ccmake) line above.
This has the same effect as the --prefix=/path/ configure option.

For further documentation read:
1) http://www.cmake.org/HTML/RunningCMake.html
2) http://www.cmake.org/HTML/Examples.html

WARNING: cmake isn't the deafut tool to compile FANN. Vincenzo Di Massa <hawk.it@tiscali.it> 
added fann support to FANN just because he likes CMake a lot :-)
Because of this the CMakeFiles.txt provided are to be considered work in progress
(but they seem to work like a charm).
Please provide feedback to the FANN-CMake maintainer hawk.it@tiscali.it using either of
1) "FANN General and development discussion" <fann-general@lists.sourceforge.net>
2) http://leenissen.dk/fann/forum/
3) "Vincenzo Di Massa" <hawk.it@tiscali.it>

WARNING #2: CMake is addictive! Once you start using it you'll have the desire to port
anything to CMake and to tell anybody about CMake :-)
