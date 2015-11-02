#!/usr/bin/env ruby
#=========================================================================
# run-latex.rb [options] -- [latex-options] texfile.tex
#=========================================================================
#
#     --latex-prog prog   Program to use for latex processing [latex]
#     --bibtex-prog prog  Program to use for bibtex processing [bibtex]
#  -I --input-dir dir     Directory where inputs are found [.]
#  -v --[no-]verbose      Verbose mode
#  -h --help              Display this message
#
# This script will run LaTex and BibTeX the minimum number of times to
# correctly compile the given tex source file. You can specify the
# exact command to use for the LaTeX and BibTeX processing, and you
# can include multiple --include-dir options to specify where
# LaTeX/BibTeX should look for their inputs. Although the
# [latex-options] are passed without modification to latex, no options
# are given to bibtex. The last item on the command line should be the
# tex file to process. The general strategy for determining how to
# optimally compile the tex source file is shown below:
#
#  - Run LaTeX
#  - If .bib files are newer than .bbl then run BibTeX
#  - If .bbl file differs from .aux file run BibTeX
#  - Rerun LaTeX until cross-references are resolved
#
# Author : Christopher Batten
# Date   : August 11, 2008
#

require 'optparse'

# Enable ruby warnings (this avoid problems with "ruby -w")
$VERBOSE = true

#-------------------------------------------------------------------------
# Command line processing
#-------------------------------------------------------------------------

def usage()
  puts ""
  File::open($0).each do |line|
    exit(1) if ( !(line =~ /^\#/) )
    puts line.gsub(/^\#/,"") if (($. == 3) || ($. > 4))
  end
end

$opts = {}
def parse_cmdline()

  # Defaults
  $opts[:latex_prog]  = "latex"
  $opts[:bibtex_prog] = "bibtex"
  $opts[:input_dirs]  = ["."]

  # Parse the command line
  OptionParser.new do |opts|
    opts.on(      "--latex-prog [prog]")  { |a| $opts[:latex_prog] = a }
    opts.on(      "--bibtex-prog [prog]") { |a| $opts[:bibtex_prog] = a }
    opts.on("-I", "--input-dir [dir]")    { |a| $opts[:input_dirs].push(a) }
    opts.on("-v", "--[no-]verbose")       { |a| $opts[:verbose] = a }
    opts.on("-h", "--help")               { usage() }
  end.parse!

  # Get the tex input source file name
  $opts[:tex_full_name] = ARGV.last() or throw OptionParser::InvalidOption

rescue
  usage()
end

#-------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------

def main()

  # --- Setup ------------------------------------------------------------

  parse_cmdline()

  # Environment variables

  texinputs = ""
  if ( ENV['TEXINPUTS'] )
   texinputs = ENV['TEXINPUTS'].gsub(/^\.:/,"").gsub(/:$/,"")
  end

  bibinputs = ""
  if ( ENV['BIBINPUTS'] )
   bibinputs = ENV['BIBINPUTS'].gsub(/^\.:/,"").gsub(/:$/,"")
  end

  bstinputs = ""
  if ( ENV['BSTINPUTS'] )
   bstinputs = ENV['BSTINPUTS'].gsub(/^\.:/,"").gsub(/:$/,"")
  end

  # General variables

  latex_bname = File::basename( $opts[:tex_full_name], ".tex" )
  latex_out   = "#{latex_bname}.out"
  latex_aux   = "#{latex_bname}.aux"
  latex_bbl   = "#{latex_bname}.bbl"
  latex_bc    = "#{latex_bname}.bc"

  input_dirs  = $opts[:input_dirs].join(":")

  latex_prog  = $opts[:latex_prog]
  latex_args  = ARGV.join(" ")
  latex_cmd   = "env TEXINPUTS=\"#{input_dirs}:#{texinputs}:\" " \
                "#{latex_prog} #{latex_args} | tee #{latex_out}"

  bibtex_out  = "#{latex_bname}-bibtex.out"

  bibtex_prog = $opts[:bibtex_prog]
  bibtex_cmd  = "env BSTINPUTS=\"#{input_dirs}:#{bstinputs}:\" " \
                "    BIBINPUTS=\"#{input_dirs}:#{bibinputs}:\" " \
                "#{bibtex_prog} #{latex_bname} | tee #{bibtex_out}"

  # --- Run LaTeX --------------------------------------------------------

  puts("*** Run LaTeX ***")
  system(latex_cmd)

  # Scan output fatal errors and unresolved cross-references

  unresolved_xref = false
  labels_multidef = false
  label_undef     = false
  citation_undef  = false

  IO.foreach(latex_out) do |line|
    case line
      when /Fatal error occurred/                  then exit(1)
      when /Rerun to get cross-references/         then unresolved_xref = true
      when /There were multiply-defined labels/    then labels_multidef = true
      when /LaTeX Warning: Reference .* undefined/ then label_undef     = true
      when /LaTeX Warning: Citation .* undefined/  then citation_undef  = true
    end
  end

  # Scan aux for bibstyle, bibdata, and citations

  aux_bibstyle  = ""
  aux_bibdata   = []
  aux_citations = []

  aux_files = [ latex_aux ]
  while ( !aux_files.empty? )
    aux_file = aux_files.shift()

    IO.foreach(aux_file) do |line|
      case line
        when /^\\@input\{(.*)\}$/   then aux_files.push($1)
        when /^\\bibstyle\{(.*)\}$/ then aux_bibstyle = $1
        when /^\\bibdata\{(.*)\}$/  then aux_bibdata  = $1.split(",")
        when /^\\citation\{(.*)\}$/ then aux_citations.push($1)
      end
    end

  end
  aux_files.uniq!
  aux_citations.uniq!

  # --- Run BibTeX -------------------------------------------------------

  need_bbl   = !aux_bibdata.empty? && !aux_citations.empty?
  bbl_exists = File::exists?(latex_bbl)

  # Determine if bibs are newer than bbl

  bbl_outdated = false
  if ( need_bbl && bbl_exists )

    # Loop through the bibdata files
    for bib in aux_bibdata
      bib_file_name = bib + ".bib"

      # Search the input dirs for the bibdata file
      for dir in $opts[:input_dirs]
        bib_full_name = "#{dir}/#{bib_file_name}"
        if ( File::exists?(bib_full_name) )
          bbl_outdated |= (File::mtime(bib_full_name) > File::mtime(latex_bbl))
        end
      end

    end

  end

  # Determine if aux and bbl are different

  bc_bibstyle = ""
  bc_bibdata  = []
  bc_bibitems = []

  if ( need_bbl && bbl_exists && !bbl_outdated )

    IO.foreach(latex_bbl) do |line|
      case line
        when /^%  bibstyle = (.*)$/ then bc_bibstyle = $1
        when /^%  bibdata  = (.*)$/ then bc_bibdata  = $1.split(",")
        when /^\\bibitem\{(.*)\}$/  then bc_bibitems.push($1)
      end
    end

    bbl_outdated |= ( aux_bibstyle  != bc_bibstyle )
    bbl_outdated |= ( aux_bibdata   != bc_bibdata  )
    bbl_outdated |= ( aux_citations != bc_bibitems )

  end

  # Run BibTeX if we need to

  if ( need_bbl && !bbl_exists || bbl_outdated )

    puts("*** Run BibTeX ***")
    exit($?.exitstatus) if !system(bibtex_cmd)
    unresolved_xref = true

    File::open(latex_bbl,"a") do |file|
      file.puts("% Extra data included by run-latex.rb")
      file.puts("%  bibstyle = #{aux_bibstyle}")
      file.puts("%  bibdata  = #{aux_bibdata.join(",")}")
    end

    # Scan output for warnings and treat as errors

    IO.foreach(bibtex_out) do |line|
      case line
        when /Warning--I didn't find a database entry for/ then dummy = true
        when /Warning/ then exit(1)
      end
    end

    # If we ran BibTeX we need to rerun LaTeX an extra time in _addition_
    # to running again to resolve unresolved cross-references.

    puts("*** Rerun LaTeX (due to running BibTeX) ***")
    system(latex_cmd)

    # Scan output fatal errors

    labels_multidef = false
    label_undef     = false
    citation_undef  = false

    IO.foreach(latex_out) do |line|
      case line
        when /Fatal error occurred/                  then exit(1)
        when /There were multiply-defined labels/    then labels_multidef = true
        when /LaTeX Warning: Reference .* undefined/ then label_undef     = true
        when /LaTeX Warning: Citation .* undefined/  then citation_undef  = true
      end
    end

  end

  # --- Rerun LaTeX ------------------------------------------------------

  while ( unresolved_xref )

    # Run LaTeX

    puts("*** Rerun LaTeX ***")
    system(latex_cmd)

    # Scan output fatal errors and unresolved cross-references

    unresolved_xref = false
    labels_multidef = false
    label_undef     = false
    citation_undef  = false

    IO.foreach(latex_out) do |line|
      case line
        when /Fatal error occurred/                  then exit(1)
        when /Rerun to get cross-references/         then unresolved_xref = true
        when /There were multiply-defined labels/    then labels_multidef = true
        when /LaTeX Warning: Reference .* undefined/ then label_undef     = true
        when /LaTeX Warning: Citation .* undefined/  then citation_undef  = true
      end
    end

  end

  # --- Print Summary ----------------------------------------------------

  puts("")

  if ( labels_multidef )
    puts(" [ FAILED ] Check for multiply-defined labels")
  else
    puts(" [ passed ] Check for multiply-defined labels")
  end

  if ( label_undef )
    puts(" [ FAILED ] Check for undefined references")
  else
    puts(" [ passed ] Check for undefined references")
  end

  if ( citation_undef )
    puts(" [ FAILED ] Check for undefined citations")
  else
    puts(" [ passed ] Check for undefined citations")
  end

  puts("")

end
main()

