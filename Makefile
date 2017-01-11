TEX=paper
.PHONY: all,clean


all: $(TEX).pdf


$(TEX).pdf: $(TEX).tex definitions.tex
	# for bibtex to work correctly we need to compile twice
	pdflatex $(TEX)
	pdflatex $(TEX)


clean:
	# remove all files except for .tex, .bib and the Makefile
	find -type f | grep -v 'tex\|bib\|Makefile\|figures' | xargs rm
