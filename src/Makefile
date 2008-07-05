# Makefile - NTG source code
# RMM, 1 Feb 02

# Location to install NTG
NTGDIR=..
NTGLIB=libntg.a

# 
CC=cc
CFLAGS=

LINT=lint
LTFLAGS=

INCFILES = ntg.h integrator.h colloc.h constraints.h cost.h matrix.h av.h

all: $(NTGLIB)

install: all
	-mkdir $(NTGDIR)/lib
	install $(NTGLIB) $(NTGDIR)/lib
	-mkdir $(NTGDIR)/include
	install $(INCFILES) $(NTGDIR)/include


libntg.a: colloc.o constraints.o cost.o integrator.o matrix.o ntg.o 
	ar rcv libntg.a colloc.o constraints.o cost.o integrator.o matrix.o \
	ntg.o 

colloc.o: colloc.c colloc.h av.h matrix.h 
constraints.o: constraints.c constraints.h matrix.h colloc.h 
cost.o: cost.c cost.h matrix.h integrator.h
integrator.o: integrator.c integrator.h matrix.h
matrix.o: matrix.c matrix.h
ntg.o: ntg.c ntg.h av.h colloc.h constraints.h cost.h matrix.h
test.o: test.c

test: test.o colloc.o
	cc -fast -L. -L$(HOME)/src/pgs/double \
	test.o colloc.o -lntg -lpgs -lM77 -lF77 \
	-lsunmath -lm

lint4:
	$(LINT) -dirout=.lint -errchk=%all -errhdr=%user\
		-errfmt=macro\
		-I$(HOME)/src/ntg/2.1\
		-Ncheck=%all\
		-Nlevel=4 \
		example4/main.c colloc.c constraints.c cost.c integrator.c matrix.c \
		ntg.c

lint5:
	$(LINT) -dirout=.lint -errchk=%all -errhdr=%user\
		-errfmt=macro\
		-I$(HOME)/src/ntg/2.1.debug\
		-Ncheck=%all\
		-Nlevel=4 \
		example5/main.c colloc.c constraints.c cost.c integrator.c matrix.c \
		ntg.c

clean:
	rm -f *.o *~
	rm -f libntg.a
