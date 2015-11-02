#=========================================================================
# Toplevel configure.ac for the Automatic LaTeX Build System
#=========================================================================
# Please read the documentation in 'albs-uguide.txt' for more details on
# how the Automatic LaTeX Build System works. For most projects, a
# writer will only need to change the project metadata listed right
# below.

#-------------------------------------------------------------------------
# Project metadata
#-------------------------------------------------------------------------

m4_define( proj_name,         [CS5220 HW2 Report])
m4_define( proj_maintainer,   [Ji Kim])
m4_define( proj_abbreviation, [cs5220-hw2-report])

#-------------------------------------------------------------------------
# Project version information
#-------------------------------------------------------------------------
# Version information is meant to be managed through a version control
# system's tags and revision numbers. In a working copy the version will
# not be defined here (you should just use the version control system's
# mechanisms). When we make a distribution then we can set the version
# here as formed by the scripts/vcs-version.sh script so that the
# distribution knows what version it came from. If you are not using
# version control then it is fine to set this directly.

m4_define( proj_version, [?])

#-------------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------------

AC_INIT(proj_name,proj_version,proj_maintainer,proj_abbreviation)
AC_CONFIG_SRCDIR([src/]AC_PACKAGE_TARNAME[.tex])
AC_CONFIG_AUX_DIR([scripts])
AC_CANONICAL_BUILD

#-------------------------------------------------------------------------
# ALBS specific program checks
#-------------------------------------------------------------------------

ALBS_PROG_INSTALL
ALBS_PROG_PDFLATEX
ALBS_PROG_BIBTEX
ALBS_PROG_RUBY

#-------------------------------------------------------------------------
# ALBS build system module list
#-------------------------------------------------------------------------

ALBS_MODULES([svg,py])

#-------------------------------------------------------------------------
# Output
#-------------------------------------------------------------------------

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

