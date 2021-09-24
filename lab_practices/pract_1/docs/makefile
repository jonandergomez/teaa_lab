
LATEX=pdflatex

all: formulas_kmeans.pdf formulas_gmm.pdf


%.pdf: %.tex
	@$(LATEX) $< ;  \
	    while ( grep "Rerun to get cross-references" \
			$(<:%.tex=%.log) >/dev/null ); do \
			$(LATEX) $< ; \
        done; 


clean:
	rm -rf *.log *.aux *.out *.snm *.nav *.toc *.vrb
	rm -rf *.class

