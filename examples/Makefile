# Makefile for NTG examples
# RMM, 1 Feb 02

NTGDIR = ..
NTGINCDIR = $(NTGDIR)/include
NTGLIBDIR = $(NTGDIR)/lib

PGMS = vanderpol

CC=gcc -I $(NTGINCDIR)
CFLAGS= -O3
LIB= -lm -lntg -lpgs -lnpsol -lg2c

vanderpol: vanderpol.o
	$(CC) $(CFLAGS) -L $(NTGLIBDIR) -o vanderpol vanderpol.o $(LIB)

clean:
	rm -f *.o coef1 *~
	rm -f $(PGMS)
