XML = fann.xml

all: html html-single dvi pdf ps rtf tex txt

html:
	jw -b html -o html $(XML)

html-single:
	jw -u -b html $(XML)

dvi:
	jw -u -b dvi $(XML)

pdf:
	jw -u -b pdf $(XML)

ps:
	jw -u -b ps $(XML)

rtf:
	jw -u -b rtf $(XML)

tex:
	jw -u -b tex $(XML)

txt:
	jw -u -b txt $(XML)

clean:
	rm -rf \
		html \
		fann.html \
		fann.dvi \
		fann.pdf \
		fann.ps \
		fann.rtf \
		fann.tex \
		fann.txt

distclean: clean
