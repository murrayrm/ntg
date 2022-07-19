# Makefile for NTG tests
# RMM, 16 Jul 2022

OBJFILES = lowlevel.so

default: $(OBJFILES)

%.so: %.o
	$(CC) $< -shared -o $@ -lm
