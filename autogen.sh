#!/bin/sh

aclocal-1.7
libtoolize --force
autoheader2.50
autoconf2.50
automake-1.7 --add-missing --copy
./configure $@
