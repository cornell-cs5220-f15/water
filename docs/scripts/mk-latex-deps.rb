#!/usr/bin/env ruby
#=========================================================================
# mk-latex-deps.rb [options] texfile.tex
#=========================================================================
#
#  -v --[no-]verbose  Verbose mode
#  -h --help          Display this message
#
# This script scans the given texfile.tex file to generate a make
# fragment which lists all of the prerequisites for building
# texfile.pdf. In addition to scanning texfile.tex, the script will
# recursively scan any files which texfile.tex includes so that the
# final dependency information is complete. The resulting make
# fragment is named texfile.d and it includes make variables for each
# type of prerequisite. All make variables are prefixed by texfile
# which allows the toplevel makefile to different rules for different
# prerequisite types (for example to generate figure pdfs or to
# generate partial builds via \includeonly). Currently the script
# scans for the following types of prerequisites:
#
#   \documentclass{class}
#   \usepackage{package}
#   \RequirePackage{package}
#   \input{tex}
#   \include{tex}
#   \includegraphics{graphic}
#   \bibliography{bib1,bib2,...}
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

  OptionParser.new do |opts|
    opts.on("-h", "--help")         { usage() }
    opts.on("-v", "--[no-]verbose") { |v| $opts[:verbose] = v }
  end.parse!
  $opts[:tex_full_name] = ARGV[0] or throw OptionParser::InvalidOption

  # full_name = path_name/file_name
  # file_name = base_name.file_ext

  $opts[:tex_path_name] = File::dirname( $opts[:tex_full_name] )
  $opts[:tex_file_name] = File::basename( $opts[:tex_full_name] )
  $opts[:tex_base_name] = File::basename( $opts[:tex_full_name], ".tex" )

rescue
  usage()
end

#-------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------

def write_header( file, char, text )
  file.puts( "#" + char*72 )
  file.puts( "# #{text}" )
  file.puts( "#" + char*72 )
end

#-------------------------------------------------------------------------
# SimplePrereq
#-------------------------------------------------------------------------

class SimplePrereq

  attr_reader :name, :make_vars, :reg_exprs, :file_ext
  attr_reader :depend_only_if_exists, :prereq_file_names
  attr_reader :prereq_for_dep_file

  def initialize( name )
    @name              = name
    @make_vars         = []
    @reg_exprs         = []
    @prereq_file_names = []
    @file_ext          = ""
    @depend_only_if_exists = false
    @prereq_for_dep_file   = false
  end

  def add_make_var( make_var )
    @make_vars.push(make_var)
  end

  def add_reg_expr( reg_expr )
    @reg_exprs.push(reg_expr)
  end

  def set_file_ext( file_ext )
    @file_ext = file_ext
  end

  def set_depend_only_if_exsts( depend_only_if_exists )
    @depend_only_if_exists = depend_only_if_exists
  end

  def set_prereq_for_dep_file( prereq_for_dep_file )
    @prereq_for_dep_file = prereq_for_dep_file
  end

  # Check to see if this line contains a prereq

  def check( line )
    prereq_full_names = []
    for reg_expr in @reg_exprs
      if ( line =~ reg_expr )
        for prereq in $1.split(",")

          prereq_file_name = prereq.strip + @file_ext
          prereq_full_name = $opts[:tex_path_name]+"/"+prereq_file_name
          prereq_exists    = File::exists?(prereq_full_name)

          # Add prereq to list of full names we will return
          if ( prereq_exists )
            prereq_full_names.push(prereq_full_name)
          end

          # Add prereq to internal list
          if ( prereq_exists || !@depend_only_if_exists )
            @prereq_file_names.push(prereq_file_name)
            @prereq_file_names.uniq!
          end

        end
      end
    end
    return prereq_full_names
  end

  # Output the prereqs to the dependency file

  def write_dep_file( dep_file )
    bname = $opts[:tex_base_name].gsub(/-/,"_")
    dep_file.puts( "# #{@name}" )

    # Primary make variable containt all prereqs of this type
    dep_file.puts( "\ndoc_#{bname}_#{@name} := \\" )
    for prereq_file_name in prereq_file_names
      dep_file.puts( "  #{prereq_file_name} \\" )
    end
    dep_file.puts( "" )

    # Add prereqs of this type to various running variables
    dep_file.puts( "#{@name} += $(doc_#{bname}_#{@name})" )
    for make_var in @make_vars
      dep_file.puts( "doc_#{bname}_#{make_var} += $(doc_#{bname}_#{@name})" )
      dep_file.puts( "#{make_var} += $(doc_#{bname}_#{@name})" )
    end
    if ( @prereq_for_dep_file )
      dep_file.puts( "doc_#{bname}_dep_prereqs += $(doc_#{bname}_#{@name})" )
    end
    dep_file.puts( "doc_#{bname}_prereqs += $(doc_#{bname}_#{@name})" )
    dep_file.puts( "" )
  end

end

#-------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------

def main()

  parse_cmdline()

  # --- Prerequisite Types -----------------------------------------------

  # To add a new kind of prerequisite, normally all you will need to
  # do is instantiate SimplePrereq, set the various options
  # appropriately, and push it onto the prereq_types list.

  prereq_types = []

  # \documentclass{class}

  documentclass = SimplePrereq::new("documentclasses")
  documentclass.add_reg_expr( /^[^%]*\\documentclass\{([^}]+)\}/ )
  documentclass.add_reg_expr( /^[^%]*\\documentclass\[.*\]\{([^}]+)\}/ )
  documentclass.set_file_ext( ".cls" )
  documentclass.set_depend_only_if_exsts(true)
  documentclass.set_prereq_for_dep_file(true)
  prereq_types.push(documentclass)

  # \usepackage{package} and \RequirePackage{package}

  package = SimplePrereq::new("packages")
  package.add_reg_expr( /^[^%]*\\usepackage\{([^}]+)\}/ )
  package.add_reg_expr( /^[^%]*\\usepackage\[.*\]\{([^}]+)\}/ )
  package.add_reg_expr( /^[^%]*\\RequirePackage\{([^}]+)\}/ )
  package.add_reg_expr( /^[^%]*\\RequirePackage\[.*\]\{([^}]+)\}/ )
  package.set_file_ext( ".sty" )
  package.set_depend_only_if_exsts(true)
  package.set_prereq_for_dep_file(true)
  prereq_types.push(package)

  # \input{tex}

  input_tex = SimplePrereq::new("input_texs")
  input_tex.add_reg_expr( /^[^%]*\\input\{([^}]+)\}/ )
  input_tex.set_file_ext( ".tex" )
  input_tex.set_prereq_for_dep_file(true)
  prereq_types.push(input_tex)

  # \include{tex}

  include_tex = SimplePrereq::new("include_texs")
  include_tex.add_reg_expr( /^[^%]*\\include\{([^}]+)\}/ )
  include_tex.set_file_ext( ".tex" )
  include_tex.set_prereq_for_dep_file(true)
  prereq_types.push(include_tex)

  # \includegraphics{graphic}

  include_graphic = SimplePrereq::new("include_graphics")
  include_graphic.add_make_var( "figs" )
  include_graphic.add_reg_expr( /^[^%]*\\includegraphics\{([^}]+)\}/ )
  include_graphic.add_reg_expr( /^[^%]*\\includegraphics\[.*\]\{([^}]+)\}/ )
  include_graphic.add_reg_expr( /^[^%]*\\includegraphics<.*>\[.*\]\{([^}]+)\}/ )
  prereq_types.push(include_graphic)

  # \bibliography{bib1,bib2,...}

  bib = SimplePrereq::new("bibs")
  bib.add_reg_expr( /^[^%]*\\bibliography\{([^}]+)\}/ )
  bib.set_file_ext( ".bib" )
  prereq_types.push(bib)

  # --- Scan for Prerequisites -------------------------------------------

  # Process list of prereqs, adding to it as we go along, until empty

  prereq_full_names = [ $opts[:tex_full_name] ]
  while ( !prereq_full_names.empty? ) do
    prereq_full_name = prereq_full_names.shift()

    # For each line in prereq file see if it includes any other prereqs

    IO.foreach(prereq_full_name) do |line|
      for prereq_type in prereq_types
        prereq_full_names.push(prereq_type.check(line))
        prereq_full_names.flatten!.uniq!
      end
    end

  end

  # --- Write the Dependency Make Fragment -------------------------------

  # Open the dependency file

  dep_file_name = $opts[:tex_base_name] + ".d"
  dep_file = File::new(dep_file_name,"w")

  # Write the top header

  write_header( dep_file, '=', dep_file_name )
  dep_file.puts("# Autogenerated by mk-latex-deps.rb")
  dep_file.puts("# Do not edit manually\n\n")

  # Output make variables for each prereq type

  write_header( dep_file, '-', "Variables for each prerequisite type"  )
  dep_file.puts("")
  for prereq_type in prereq_types
    prereq_type.write_dep_file(dep_file)
  end

  # Output the main dependency rule and empty rules for each prereq to
  # avoid wedging make when one of the prereqs does not exist

  write_header( dep_file, '-', "Dependency rules"  )

  pdf_file_name = $opts[:tex_base_name] + ".pdf"
  tex_file_name = $opts[:tex_file_name]

  tex_base_name = $opts[:tex_base_name].gsub(/-/,"_")
  prereqs_make_var     = "$(doc_#{tex_base_name}_prereqs)"
  dep_prereqs_make_var = "$(doc_#{tex_base_name}_dep_prereqs)"

  dep_file.puts("")
  dep_file.puts("#{pdf_file_name} : #{tex_file_name} #{prereqs_make_var}")
  dep_file.puts("#{dep_file_name} : #{tex_file_name} #{dep_prereqs_make_var}")

  for prereq_type in prereq_types
    for prereq_file_name in prereq_type.prereq_file_names
      dep_file.puts("\n#{prereq_file_name} : ")
    end
  end

  # Close the dependency file

  dep_file.close()

end
main()

