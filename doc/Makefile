XML = fann.xml
BACKENDS = dvi html pdf ps rtf tex txt

all:
	jw -b dvi -o dvi $(XML)
	jw -b html -o html $(XML)
	jw -b pdf -o pdf $(XML)
	jw -b ps -o ps $(XML)
	jw -b rtf -o rtf $(XML)
	jw -b tex -o tex $(XML)
	jw -b txt -o txt $(XML)
%:
	jw -b $@ -o $@ $(XML)
clean:
	rm -rf $(BACKENDS)
