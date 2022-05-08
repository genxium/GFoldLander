#!/bin/bash

rm ./*.aux
rm ./*.xml
rm ./*.gz
rm ./*.bbl
rm ./*.blg
pdflatex the_paper.tex
bibtex the_paper
pdflatex the_paper.tex
rm ./*.aux
rm ./*.xml
rm ./*.gz
rm ./*.bbl
rm ./*.blg

rm ./*.out
