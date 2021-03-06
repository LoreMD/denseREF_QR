#------------------------------------------------------------------------------
# SPEX_QR/Makefile: compile, install, and run SLIP QR demo and tests
#------------------------------------------------------------------------------

# SPEX_QR: (c) 2021, Chris Lourenco, US Naval Academy, All Rights Reserved.
# SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

#------------------------------------------------------------------------------

# default: compile the dynamic library, and run the demos
default: C

include ./SPEX-1.1.2/SuiteSparse_config/SuiteSparse_config.mk

#PATH_TO_SPEX=-Wl,-rpath,"./SPEX-1.1.2/lib/"
#CF = $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) $(PATH_TO_SPEX) -O
#I = -I./Include -I./Source -I./SPEX-1.1.2/SuiteSparse_config -I./SPEX-1.1.2/COLAMD/Include -I./SPEX-1.1.2/AMD/Include -I./SPEX-1.1.2/SPEX/SPEX_Util/Include -I./SPEX-1.1.2/SPEX/SPEX_Util/Source -I./SPEX-1.1.2/SPEX/SPEX_Util/Lib -I./SPEX-1.1.2/SPEX/SPEX_Util/Include/

# compile the dynamic library, and then run the demos and statement coverage
demos: all

# compile the dynamic library only, and then run the demos
C:
	( cd SPEX-1.1.2 ; $(MAKE) )
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

# compile the dynamic library, and then run the demos and statement coverage
all: C cov

# compile the dynamic library only
library:
	( cd Lib ; $(MAKE) )

# compile the static library only
static:
	( cd Lib ; $(MAKE) static )

# statement coverage test
cov:
	( cd Tcov ; $(MAKE) )

# remove files not in the distribution, but keep compiled libraries
clean:
	( cd Lib ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )
	( cd Tcov ; $(MAKE) clean )
	- ( cd MATLAB/Source ; $(RM) *.o )
	- ( cd MATLAB    ; $(RM) *.o )
#	( cd Doc ; $(MAKE) clean )

# remove all files not in the distribution
purge:
	( cd Lib ; $(MAKE) purge )
	( cd Demo ; $(MAKE) purge )
	( cd Tcov ; $(MAKE) purge )
	- ( cd MATLAB/Source ; $(RM) *.o *.mex* )
	- ( cd MATLAB    ; $(RM) *.o *.mex* )
#	( cd Doc ; $(MAKE) clean )

distclean: purge

# install the library
install: library
	( cd Lib ; $(MAKE) install )

# uninstall the library
uninstall:
	( cd Lib ; $(MAKE) uninstall )

