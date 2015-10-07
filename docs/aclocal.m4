#=========================================================================
# Local Autoconf Macros
#=========================================================================
# This file contains the macros for the automatic LaTeX build system and
# additional autoconf macros which developers can use in their
# configure.ac scripts. Please read the documentation in
# 'albs-uguide.txt' for more details on how the Automatic LaTeX Build
# System works. The documenation for each macro should include
# information about the author, date, and copyright.

#-------------------------------------------------------------------------
# ALBS_PROG_INSTALL
#-------------------------------------------------------------------------
# This macro will add an --enable-stow command line option to the
# configure script. When enabled, this macro will first check to see if
# the stow program is available and if so it will set the $stow shell
# variable to the binary name and the $enable_stow shell variable to
# "yes". These variables are used in the MCPPBS makefile to
# conditionally use stow for installation. This macro will also check to
# see if the $STOW_PREFIX environment variable is set, and if so it will
# change the prefix to this environment variable (assuming --prefix was
# not set on the command line). This allows a user to specify once were
# all packages should be installed.

AC_DEFUN([ALBS_PROG_INSTALL],
[
  AC_PROG_INSTALL
   
  AC_ARG_VAR([STOW_PREFIX],[Prefix for stow-based installs])
  AC_ARG_ENABLE(stow,
    AS_HELP_STRING(--enable-stow,[Enable stow-based install]),
      [enable_stow="yes"],[enable_stow="no"])
   
  AS_IF([ test "${enable_stow}" = "yes" ],
  [
    AC_CHECK_PROGS([stow],[stow],[no])  
    AS_IF([ test "${stow}" = "no" ],
    [
      AC_MSG_ERROR([Cannot use --enable-stow since stow is not available])
    ])
   
    AS_IF([ test "${prefix}" = "NONE" && test -n "${STOW_PREFIX}" ],
    [
      prefix="${STOW_PREFIX}"
      AC_MSG_NOTICE([Using \$STOW_PREFIX from environment])
      AC_MSG_NOTICE([prefix=${prefix}])
    ])
   
  ])

  AC_SUBST([enable_stow])
])

#-------------------------------------------------------------------------
# ALBS_PROG_PDFLATEX
#-------------------------------------------------------------------------
# Checks to make sure that pdflatex is in users path otherwise the
# configuration fails.

AC_DEFUN([ALBS_PROG_PDFLATEX],
[
  AC_CHECK_PROGS([pdflatex],[pdflatex],[no])
  AS_IF([test $pdflatex = "no"],
  [
    AC_MSG_ERROR([Automatic LaTeX Build System requires PDFLaTeX])
  ])
])

#-------------------------------------------------------------------------
# ALBS_PROG_BIBTEX
#-------------------------------------------------------------------------
# Checks to make sure that bibtex is in users path otherwise the
# configuration fails. Technically, we don't need BibTeX if we are not
# going to use a bibliography, but since pdflatex almost always comes
# with bibtex we stop if we cannot find bibtex since it means 
# something is probably setup wrong.

AC_DEFUN([ALBS_PROG_BIBTEX],
[
  AC_CHECK_PROGS([bibtex],[bibtex],[no])
  AS_IF([test $bibtex = "no"],
  [
    AC_MSG_ERROR([Automatic LaTeX Build System requires BibTeX])
  ])
])

#-------------------------------------------------------------------------
# ALBS_PROG_RUBY
#-------------------------------------------------------------------------
# Checks to make sure that ruby is in users path otherwise the
# configuration fails. We use ruby to scan files, process latex
# dependencies, and to actually control running LaTeX/BibTeX.

AC_DEFUN([ALBS_PROG_RUBY],
[
  AC_CHECK_PROGS([ruby],[ruby],[no])
  AS_IF([test $ruby = "no"],
  [
    AC_MSG_ERROR([Automatic LaTeX Build System requires ruby])
  ])
])

#-------------------------------------------------------------------------
# ALBS_MODULES
#-------------------------------------------------------------------------
# Used to specify a list of modules to use for this document. The list
# can include whitespace and newlines for readability.

AC_DEFUN([ALBS_MODULES],
[

  # Add command line option to disable all modules

  AC_ARG_WITH(modules,
    AS_HELP_STRING([--without-modules],[Disable all modules]),
    [with_modules="no"],
    [with_modules="yes"])

  # Loop through the modules given in the macro argument

  m4_foreach([ALBS_MODULE],[$1],
  [

    # Create variations of the module name 

    m4_define([ALBS_MODULE_NORM],m4_normalize(ALBS_MODULE))
    m4_define([ALBS_MODULE_SHVAR_WITH],with_[]ALBS_MODULE_NORM)
    m4_define([ALBS_MODULE_SHVAR_EN],ALBS_MODULE_NORM[]_enabled)

    # Add command line option to disable module

    AC_ARG_WITH(ALBS_MODULE_NORM, 
      AS_HELP_STRING([--without-ALBS_MODULE_NORM],
        [Disable the ALBS_MODULE_NORM module]),
      [ALBS_MODULE_SHVAR_WITH="no"],
      [ALBS_MODULE_SHVAR_WITH="yes"])

    # Add module to our running list

    modules="$modules ALBS_MODULE_NORM"

    # For each module include the appropriate autoconf fragment

    AS_IF([test    "$ALBS_MODULE_SHVAR_WITH" = "yes" \
              -a "$with_modules" = "yes" ],
    [
      AC_MSG_NOTICE([configuring module : ALBS_MODULE_NORM])
      m4_include(ALBS_MODULE_NORM[]/ALBS_MODULE_NORM[].ac) 
    ],[
      AC_MSG_NOTICE([skipping module : ALBS_MODULE_NORM])
      ALBS_MODULE_SHVAR_EN="no"
    ])

  # Tell autoconf about the module's .mk.in file

    AC_CONFIG_FILES(ALBS_MODULE_NORM[].mk:ALBS_MODULE_NORM[]/ALBS_MODULE_NORM[].mk.in)

  # Substitute the module_enable make variable

    AC_SUBST(ALBS_MODULE_SHVAR_EN)

  ])

  # Output make variables

  AC_SUBST([modules])

])
