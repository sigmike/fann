SUBDIRS = include

lib_LTLIBRARIES = libfloatfann.la libdoublefann.la libfixedfann.la libfann.la

AM_LDFLAGS = -version-info 1:5:0
AM_CFLAGS = -O9 -ansi

libfloatfann_la_SOURCES = floatfann.c
libdoublefann_la_SOURCES = doublefann.c
libfixedfann_la_SOURCES = fixedfann.c
libfann_la_SOURCES = fann.c fann_internal.c
