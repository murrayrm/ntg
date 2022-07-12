# Makefile for NTG examples
# RMM, 1 Feb 02

NTGDIR = ..
NTGINCDIR = $(NTGDIR)/include
NTGLIBDIR = $(NTGDIR)/lib

PGMS = vanderpol kincar

CFLAGS = -g
CPPFLAGS += -I $(NTGINCDIR) 
LIB = -lm -lntg -lpgs -lnpsol -lgfortran

vanderpol: vanderpol.o $(NTGLIBDIR)/*.a
	$(CC) $(CFLAGS) $(CPPFLAGS) -L $(NTGLIBDIR) -o $@ vanderpol.o $(LIB)

kincar: kincar.o $(NTGLIBDIR)/*.a
	$(CC) $(CFLAGS) $(CPPFLAGS) -L $(NTGLIBDIR) -o $@ kincar.o $(LIB)

%.so: %.o
	$(CC) $< -shared -L $(NTGLIBDIR) -o $@ $(LIB)



clean:
	rm -f *.o coef1 *~
	rm -f $(PGMS)
