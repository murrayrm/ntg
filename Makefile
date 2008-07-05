# Makefile - main makefile for NTG software package
# RMM, 2 Feb 02

# Desired installation location of main NTG directory
# 
# By default, this is set to the current director.  This is useful
# if you are a developer, but probably not what you want to use if
# you are installing it for others to use.
#
NTGDIR = $(CURDIR)
# NTGDIR = /usr/local/ntg

# Files and directories to include in distribution
FILES = README Makefile Pending
DIRS  = src/ doc/ examples/ pgs/

# Default rule: make everything in the subdirectories
#
# Note that pgs and npsol are not distributed with NTG, so it's OK
# if these are not in place
#
all:
	(cd src; make)

# Install all of the required packages
install:
	-mkdir $(NTGDIR)/lib
	-mkdir $(NTGDIR)/include
	(cd src; make NTGDIR=$(NTGDIR) install)

# Generate a tar file for distribution to others
tar: clean
	tar cf ntg.tar $(FILES) $(DIRS)

clean:
	(cd src; make clean)
	(cd examples; make clean)
	(cd pgs; make clean)
	-(cd doc; make clean)

tidy:
	rm *~
