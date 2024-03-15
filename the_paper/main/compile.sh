#!/bin/bash

function clearRedundancies {
	rm ./*.aux
	rm ./*.spl
	rm ./*.xml
	rm ./*.gz
	rm ./*.bbl
	rm ./*.blg
	rm ./*.out
	rm ./*.log
}

rm ./*.pdf
clearRedundancies
pdflatex manuscript.tex
bibtex manuscript.aux
pdflatex manuscript.tex
pdflatex manuscript.tex
clearRedundancies
