SUBDIRS = src

EXTRA_DIST = doc examples benchmarks

dist-hook:
	rm -rf `find $(distdir)/doc -name CVS`
	rm -rf `find $(distdir)/examples -name CVS`
	rm -rf `find $(distdir)/benchmarks -name CVS`
