XML = fann.xml

all: html html-single dvi pdf ps rtf tex txt

html: fann.xml
	jw -b html -o html $(XML)

html-single: fann.xml
	jw -u -b html $(XML)

dvi: fann.xml
	jw -u -b dvi $(XML)

pdf: fann.xml
	jw -u -b pdf $(XML)

ps: fann.xml
	jw -u -b ps $(XML)

rtf: fann.xml
	jw -u -b rtf $(XML)

tex: fann.xml
	jw -u -b tex $(XML)

txt: fann.xml
	jw -u -b txt $(XML)

clean:
	rm -rf \
		html html-single dvi pdf ps rtf tex txt \
		fann.html \
		fann.dvi \
		fann.pdf \
		fann.ps \
		fann.rtf \
		fann.tex \
		fann.txt

distclean: clean
