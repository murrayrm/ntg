# Makefile for NPSOL
# RMM, 1 Feb 02 based on installed copy

LIBDIR	= ../lib

NPSRC   = chsubs.f   srsubs.f  npsubs.f
LSSRC   = mcsubs.f   opsubs.f  blas1.f      blas2.f   \
          f06subs.f  cmsubs.f  qrsubs.f     lssubs.f

libnpsol.a: $(NPSRC:.f=.o) $(LSSRC:.f=.o)
	ar ruv $@ *.o
	ranlib $@

install:
	mkdir -p $(LIBDIR)
	install libnpsol.a $(LIBDIR)

test: npmain
npmain: npmain.o
	$(F77) $(FFLAGS) -o $@ $< -L . -lnpsol

.SUFFIXES: .f .o
.f.o:
	$(F77) $(FFLAGS) -c $*.f

clean:
	rm -f npmain *.o *~ 
	rm -f libnpsol.a

distclean: clean
	rm -f *.f npsol.* 


