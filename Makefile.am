SUBDIRS = src

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = fann.pc

EXTRA_DIST = doc examples benchmarks

dist-hook:
	rm -rf `find $(distdir) -name "*~" -o -name ".#*"`
	rm -rf `find $(distdir)/doc -name CVS`
	rm -rf `find $(distdir)/examples -name CVS`
	rm -rf `find $(distdir)/benchmarks -name CVS`
	zip -r $(distdir).zip $(distdir)
	tar chojf $(distdir).tar.bz2 $(distdir)
