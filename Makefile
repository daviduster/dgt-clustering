PDFLATEX:=pdflatex
MAINTEX=P2-Martín-Vela-DavidAlberto.tex
OUTPUT_DIR=../
MAIN=P2-Martín-Vela-DavidAlberto

all:
	cd doc/ && pdflatex $(MAINTEX) > /dev/null && bibtex $(MAIN) > /dev/null && pdflatex $(MAINTEX) > /dev/null && pdflatex -output-directory=$(OUTPUT_DIR) $(MAINTEX) > /dev/null && cd $(OUTPUT_DIR) && make clean

clean: 
	@rm -f *aux *xml *bbl *blg *toc *log *out *lof *lot $(MAIN)-blx.bib

view:
	open $(MAIN).pdf
 