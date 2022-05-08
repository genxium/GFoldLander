Latex Dependencies from TeXLive 
--

- Installing Binaries
    - On Mac OSX, visit [the MacTeX site](http://www.tug.org/mactex/), download and install either MacTeX or BasicTex package.
    - On Debian/Ubuntu, there're 2 options.
        - Due to historical compatibility issues, If you want to use `tlmgr` to manage TeX packages later, please only [download the .tar.gz package from here](http://tug.org/texlive/acquire-netinstall.html) and [follow the instructions here](http://tug.org/texlive/quickinstall.html) for installation, i.e. DON'T use `apt`!
        - If you're OK without `tlmgr`, please follow [the TeXLive for Debian/Ubuntu documentation to install](https://www.tug.org/texlive/debian.html).
- Converting LaTeX Scripts to PDF
    - After installed, you should be able to use binary such as `pdflatex` to convert your `*.tex` files into `*.pdf`.
- [TeXLive Manager](http://tug.org/texlive/tlmgr.html)
    - After following the instruction above, you've also got a binary `tlmgr` which allows you to manage TeXLive style packages.
        - You'll be in need of this binary when for example an extra formatter `titlesec` is preferred, and
        ```
        user@somewhere> tlmgr install titlesec -v
        ```
        will do the trick.
    - If you unfortunately have difficulty using `tlmgr`, e.g. on Debian/Ubuntu, please try
    ```
    user@somewhere> sudo apt-get install texlive-formats-extra --fix-missing
    ```
    which provides a bunch of common TeXLive style packages including `titlesec`.

Python Dependencies for Matplot and SchemDraw
--

- python 2.7 or 3.x
- pip 2.7 or 3.x
- pip install matplotlib
- pip install SchemDraw 


