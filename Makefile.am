SUBDIRS = src

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = fann.pc

EXTRA_DIST = benchmarks doc/fann.xml doc/fann_doc_complete_1.0.pdf doc/Makefile doc/style.css examples python MSVC++ src/include/config.h \
	debian/changelog debian/compat debian/control debian/copyright \
	debian/docs debian/libfann1-dev.dirs debian/libfann1-dev.examples \
	debian/libfann1-dev.files debian/libfann1-dev.install \
	debian/libfann1.dirs debian/libfann1.files debian/libfann1.install \
	debian/rules 

dist-hook:
	rm -rf `find $(distdir) -name "*~" -o -name ".#*"`
	(cd $(distdir)/doc && make clean && make html)
	rm -rf `find $(distdir)/doc -name CVS`
	(cd $(distdir)/examples && make clean)
	rm -rf `find $(distdir)/examples -name CVS`
	(cd $(distdir)/benchmarks && make clean)
	rm -rf `find $(distdir)/benchmarks -name CVS`
	(cd $(distdir)/python && make clean)
	rm -rf `find $(distdir)/python -name CVS`
	zip -9 -r $(distdir).zip $(distdir)
	tar chojf $(distdir).tar.bz2 $(distdir)
