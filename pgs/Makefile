# Makefile for NPSOL
# RMM, 1 Feb 02 based on installed copy

PGSURL = http://pages.cs.wisc.edu/~deboor/pgs
LIBDIR	= ../lib
FFLAGS = -O3

SRC = banfac.f banslv.f bchfac.f bchslv.f bsplpp.f bsplvb.f bsplvd.f \
  bspp2d.f bvalue.f chol1d.f colpnt.f cubspl.f difequ.f eqblok.f \
  interv.f knots.f l2appr.f l2err.f l2knts.f l2main.f newnot.f \
  newnotfk.f ppvalu.f putit.f round.f setdatx2.f setdatx3.f setdatx4.f \
  setupq.f smooth.f spli2d.f splint.f

OBJ = $(SRC:.f=.o)

libpgs.a: $(OBJ)
	ar rcv $@ $?

install: libpgs.a
	-mkdir $(LIBDIR)
	install libpgs.a $(LIBDIR)

clean:
	rm -f $(OBJ) *~
	rm -f libpgs.a

# Create a rule to try to get the PGS source code if it is not present
%.f:;	wget $(PGSURL)/$@
