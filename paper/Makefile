.PHONY: all main thesis supplement check
LATEXMK ?= latexmk
export TEXINPUTS := ./vendor//:$(TEXINPUTS)
export BSTINPUTS := ./vendor//:$(BSTINPUTS)
all: main thesis supplement
main thesis supplement:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build $@.tex
check:
	bash scripts/check_build.sh
