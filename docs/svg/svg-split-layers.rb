#!/usr/bin/env ruby
#=========================================================================
# svg-layer-split.rb [options] layer1 layer2 ... svgfile.svg
#=========================================================================
#
#  -h --help          Display this message
#  -v --[no-]verbose  Verbose mode
#
# Extract given layers into a new SVG file called
# svgfile-layer1-layer2.svg
#
# Author : Christopher Batten
# Date   : August 17, 2008
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

  $opts[:layer_names] = []

  OptionParser.new do |opts|
    opts.on("-v", "--[no-]verbose") { |v| $opts[:verbose] = v }
    opts.on("-h", "--help")         { usage() }
  end.parse!

  $opts[:svg_in_full_name] = ARGV.pop()
  $opts[:layer_names]      = ARGV

rescue
  usage()
end

#-------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------

def main()
  parse_cmdline()

  # Output file name

  svg_in_base_name    = File::basename($opts[:svg_in_full_name],".svg")
  svg_out_layer_names = $opts[:layer_names].join("-")
  svg_out_file_name   = "#{svg_in_base_name}-#{svg_out_layer_names}.svg"

  # Collect svg header

  group_opened   = false
  group_is_layer = false
  group_name     = ""
  group_style    = ""

  svg_out_file = File::new(svg_out_file_name,"w")
  IO.foreach($opts[:svg_in_full_name]) do |line|

    if ( group_opened )
      line =~ /^\s*(.+)="(.+)">?\s*/ or exit(1)
      key = $1
      val = $2

      if ( key == "inkscape:label" )
        group_name = val
      end

      if ((key == "inkscape:groupmode") && (val == "layer"))
        group_is_layer = true
      end

      if ( key != "style" )
        if ( group_is_layer && (line =~ /(.*)>$/) )
          svg_out_file.puts($1)
        else
          svg_out_file.puts(line)
        end
      else
        group_style = val
      end

      if ( line =~ />$/ )
        group_opened = false
        if ( group_is_layer )
          if ( $opts[:layer_names].include?(group_name) )
            svg_out_file.puts("style=\"display:inline\">")
          else
            svg_out_file.puts("style=\"display:none\">")
          end
        else
          svg_out_file.puts("style=\"#{group_style}\">")
        end
      end

    else
      svg_out_file.puts(line)
    end

    if ( line =~ /^\s*<g\s*$/ )
      group_opened   = true
      group_is_layer = false
      group_id       = ""
      group_style    = ""
    end

  end
  svg_out_file.close()

end
main()

