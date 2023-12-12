#!/bin/env ruby

require 'yaml'
require './darknet'

def convert(cfgfile, ymlfile)
  cfg2 = Darknet.load_config(cfgfile)

  model = { model: { layers: [] } }

  cfg2.each do |section, i|
    name = section.name
    if name == "convolutional"
      name = "conv" # pixienn shortens
    end

    model[:model][:layers].append({ type: name }) unless i == 0

    section.each do |key, value|
      if i == 0
        model[:model][key] = value
      else
        model[:model][:layers][i - 1][key] = value
      end
    end
  end

  File.open(ymlfile, "w") { |file| file.write(model.to_yaml) }
end

def usage
  $stderr.puts("usage: #{File.basename($0)} darknet-cfg outfile-yaml")
  exit(1)
end

if ARGV.length < 2
  usage
end

begin
  convert(ARGV[0], ARGV[1])
rescue StandardError => e
  eputs e, e.backtrace
  exit(1)
end


