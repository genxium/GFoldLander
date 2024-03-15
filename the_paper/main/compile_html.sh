#!/bin/bash
# Reference1 https://www.tug.org/tex4ht/doc/mn-commands.html
# Reference2 https://github.com/leeper/htlatex-example
# Reference3 https://kozyakin.github.io/latex2html/latex2html_via_tex4ht.html (for make4htconf)

function clearRedundancies {
        rm ./*.aux
        rm ./*.spl
        rm ./*.xml
        rm ./*.gz
        rm ./*.bbl
        rm ./*.blg
        rm ./*.out
        rm ./*.log
        rm ./*.tmp
        rm ./*.xref
        rm ./*.lg
        rm ./*.idv
        rm ./*.dvi
}

rm ./*.html
rm ./*.htm
rm ./manuscript*.png
rm ./*.svg
rm ./*.css
rm ./*.4ct
rm ./*.4tc
rm ./cmr*.png
rm ./cms*.png
clearRedundancies
MATH_HT_LIB=mathjax
#MATH_HT_LIB=mathml
#MATH_HT_LIB= 
mk4cfg=make4htconf.cfg

function mk4cmd {
    make4ht -sc $mk4cfg manuscript "0,$MATH_HT_LIB,p-indent,charset=utf-8" " -cunihtf -utf8"
    #make4ht manuscript "html,html5,charset=utf-8,svg" "-cmozhtf -utf8"
}

function mk4DirectlyToOdt {
    make4ht -e make4htconf_odt.mk -uf odt manuscript.tex
}

mk4cmd
bibtex manuscript
mk4cmd
#pandoc -f html+tex_math_dollars+tex_math_single_backslash --mathjax manuscript.html -o manuscript-htlatex.docx # Only enable this line if you have "pandoc" installed 
#pandoc -f html+tex_math_dollars+tex_math_single_backslash --mathjax manuscript.html -o manuscript-htlatex.odt # Only enable this line if you have "pandoc" installed 
clearRedundancies
