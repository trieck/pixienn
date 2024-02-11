#!/bin/env ruby

image_directory = '/opt/imagenet/train/images'
labels_file = '/home/trieck/src/pixienn/resources/data/imagenet.labels.list'

# Read labels from the imagenet_labels.list file
valid_labels = File.readlines(labels_file, chomp: true)

Dir.glob(File.join(image_directory, '*.JPEG')).each do |image_path|
  # Extract the prefix from the filename
  prefix = File.basename(image_path, File.extname(image_path)).split('_').first

  # Check if the prefix is in the list of valid labels
  unless valid_labels.include?(prefix)
    puts "Deleting #{image_path} as it doesn't match any valid label."
    File.delete(image_path) # Uncomment this line to actually delete the file
  end
end
