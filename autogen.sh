#!/bin/sh

aclocal
libtoolize --force
autoheader
autoconf
automake --add-missing --copy
./configure $@
