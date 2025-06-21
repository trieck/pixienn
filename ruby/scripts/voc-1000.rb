require 'pathname'

# Path to JPEG images
image_dir = '/opt/VOCdevkit/VOC2007/JPEGImages'

# Collect all JPEG filenames
images = Dir["#{image_dir}/*.jpg"].map { |f| Pathname(f).realpath.to_s }

# Sanity check
raise "Not enough images (found #{images.size})" if images.size < 1000

# Randomly select 1000
sampled = images.sample(1000)

# Output to train-1000.txt (full paths)
File.open('val-1000.txt', 'w') do |f|
  sampled.each { |line| f.puts line }
end

puts "Wrote val-1000.txt with #{sampled.size} entries."
