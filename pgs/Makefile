# Makefile for NPSOL
# RMM, 1 Feb 02 based on installed copy

# Flags to allow compilation using gfortran

PGSURL = http://pages.cs.wisc.edu/~deboor/pgs
LIBDIR	= ../lib

# Source code required for NTG
SRC = banfac.f banslv.f bchfac.f bchslv.f bsplpp.f bsplvb.f bsplvb.f	\
  bsplvd.f bspp2d.f bvalue.f chol1d.f colloc.f colpnt.f cubspl.f	\
  cwidth.f difequ.f eqblok.f interv.f knots.f l2appr.f l2err.f		\
  l2knts.f l2main.f newnot.f newnotfk.f ppvalu.f putit.f		\
  reinsch.f round.f setdatx2.f setdatx3.f setdatx4.f setupq.f		\
  slvblktx.f smooth.f spli2d.f splint.f splopt.f tautsp.f titand.f

# Examples from de Boor's book (used as tests)
EXAMPLES = iiex.f ivex.f ixex.f xex1.f xex2.f xex3.f xex4.f xiiex2.f	\
  xiiex3.f xiiex4.f xiiiex1.f xiiiex2.f xiiiex2m.f xiiiex3.f xivex1.f	\
  reinsch.f xvex.f xviex2.f xviex3.f xviiex2.f xviiex3.f

DATAFILES = xiiex2.dt1 xiiex2.dt2 xiiex3.dt1 xiiex3.dt2 xiiex3.dt3	\
  xiiex4.dt1 xiiex4.dt2 xiiiex1.dt1 xiiiex1.dt2 xiiiex2.dt1		\
  xiiiex2.dt2 xiiiex2.dt3 xiiiex2m.dt1 xiiiex2m.dt2 xiiiex2m.dt3	\
  xivex1.dt xivex2.dt xivex3.dt xivex4.dt xviex2.dt

OBJ = $(SRC:.f=.o)

libpgs.a: $(OBJ)
	ar rcv $@ $?
	ranlib $@

install: libpgs.a
	mkdir -p $(LIBDIR)
	install libpgs.a $(LIBDIR)

src: $(SRC)

test: $(EXAMPLES:.f=) $(DATAFILES)
	@for i in $(EXAMPLES:.f=); do \
	  echo "Executing $$i: (output in $$i.out)"; \
	  /bin/rm -f $$i.out; touch $$i.out; \
	  if ( [ -e $$i.dt ] || [ -e $$i.dt1 ] ); then \
	    for file in $$i.dt*; do \
	      ./$$i < $$file >> $$i.out; \
	    done; \
	  else \
	    ./$$i >> $$i.out; \
	  fi; \
	done;

# Rule to compile all of the example programs
$(EXAMPLES:.f=): $(EXAMPLES)
	@for i in $(EXAMPLES:.f=); do \
	  echo "Compiling $$i:"; \
	  $(F77) -o $$i $$i.f -L . -lpgs; \
	done;

# Grab all of the datafiles from the server
$(DATAFILES):
	@for i in $(DATAFILES); do \
	  wget $(PGSURL)/$$i; \
	done;
	touch .datafiles

clean:
	rm -f $(OBJ) *.out $(EXAMPLES:.f=) *~
	rm -f libpgs.a

distclean: clean
	rm -f *.f *.dt* .datafiles

# Create a rule to try to get the PGS source code if it is not present
%.f:;	wget $(PGSURL)/$@

.f.o:
	$(F77) $(FFLAGS) -c $*.f
