SUBDIRS = src

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = fann.pc

EXTRA_DIST = benchmarks doc examples python MSVC++

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
	zip -r $(distdir).zip $(distdir)
	tar chojf $(distdir).tar.bz2 $(distdir)
