#!/bin/env ruby

def compare_tensors(file1, file2, tolerance)
  tensor1_values = File.read(file1).split.map(&:to_f)
  tensor2_values = File.read(file2).split.map(&:to_f)

  tensor1_values.each_with_index do |tensor1_value, index|
    tensor2_value = tensor2_values[index]

    # Compare with tolerance
    if (tensor1_value - tensor2_value).abs > tolerance
      puts "Difference exceeded tolerance at position #{index + 1}:"
      puts "Tensor1 Value: #{tensor1_value}"
      puts "Tensor2 Value: #{tensor2_value}"
      puts "-----"
    end
  end

end

if ARGV.length < 2
  $stderr.puts("usage: #{File.basename($0)} file1 file2")
  exit(1)
end

tolerance = 1e-4 # Adjust the tolerance based on your requirements

compare_tensors(ARGV[0], ARGV[1], tolerance)
