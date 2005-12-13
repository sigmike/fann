SUBDIRS = src

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = fann.pc

EXTRA_DIST = benchmarks examples MSVC++ vs.net2003 src/include/config.h \
	debian/changelog debian/compat debian/control debian/copyright \
	debian/docs debian/libfann2-dev.dirs debian/libfann2-dev.examples \
	debian/libfann2-dev.files debian/libfann2-dev.install \
	debian/libfann2.dirs debian/libfann2.files debian/libfann2.install \
	debian/libfann2-py.dirs  debian/libfann2-py.files \
	debian/libfann2-py.install debian/rules win32_dll/makefile \
	win32_dll/readme.txt win32_dll/win32.ncb \
	win32_dll/win32.sln win32_dll/win32.suo win32_dll/windll.vcproj \
	win32_dll/examples/examples.ncb win32_dll/examples/examples.sln \
	win32_dll/examples/examples.suo win32_dll/examples/simple_test.vcproj \
	win32_dll/examples/simple_train.vcproj win32_dll/examples/xor_test.vcproj \
	win32_dll/examples/xor_train.vcproj \
	doc/advancedusage.txt doc/gettingstarted.txt doc/theory.txt \
	doc/fixedpointusage.txt doc/installation.txt doc/fann_en.pdf \
	python

dist-hook:
	rm -rf `find $(distdir) -name "*~" -o -name ".#*"`
	(cd $(distdir)/examples && make clean)
	rm -rf `find $(distdir)/examples -name CVS`
	(cd $(distdir)/python && make clean)
	rm -rf `find $(distdir)/python -name CVS`
	(cd $(distdir)/benchmarks && make clean)
	rm -rf `find $(distdir)/benchmarks -name CVS`
	rm -rf `find $(distdir)/MSVC++ -name CVS`
	rm -rf `find $(distdir)/vs.net2003 -name CVS`
	zip -9 -r $(distdir).zip $(distdir)
	tar chojf $(distdir).tar.bz2 $(distdir)
