require 'mini_magick'
require_relative '../utils/darknet'

label_file = '/home/trieck/src/pixienn/resources/data/imagenet.labels.list'
labels = File.readlines(label_file, chomp: true)

# Function to extract label from filename
def extract_label(filename)
  File.basename(filename, File.extname(filename)).split('_').first
end

# Function to generate darknet format entry
def generate_darknet_entry(image_path, label_index, image_size)
  w = image_size[:width]
  h = image_size[:height]

  box = image_to_darknet([w, h], [0, w, 0, h])

  sprintf("%d %.6f %.6f %.6f %.6f", label_index, *box)
end

# Directory containing ImageNet images
image_directory = '/opt/Imagenet/val'

label_output_directory = File.join(image_directory, 'labels')
Dir.mkdir(label_output_directory) unless Dir.exist?(label_output_directory)

# Iterate through each image in the directory
Dir.glob(File.join(image_directory, '*.JPEG')).each do |image_path|
  # Extract label from the filename
  label = extract_label(image_path)

  # Get label index from imagenet_labels.list
  label_index = labels.index(label)

  # Skip if label not found in the list
  next if label_index.nil?

  # Get image size
  image = MiniMagick::Image.open(image_path)
  image_size = { width: image.width, height: image.height }

  # Generate darknet format entry
  darknet_entry = generate_darknet_entry(image_path, label_index, image_size)

  label_filename = File.basename(image_path, File.extname(image_path)) + '.txt'

  label_file_path = File.join(label_output_directory, label_filename)

  File.write(label_file_path, darknet_entry)
end

