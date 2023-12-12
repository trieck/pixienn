#!/bin/env ruby

GRID_SIZE = 7

# Function to compare floating-point numbers with a specified error threshold
def float_equal?(a, b, epsilon = 1e-4)
  (a - b).abs < epsilon
end

# Function to determine grid row and column for Darknet coordinates
def darknet_to_grid(darknet_coords, grid_size)
  x, y, _, _ = darknet_coords

  # Calculate grid row and column
  grid_row = (y * grid_size).to_i
  grid_col = (x * grid_size).to_i

  [grid_row, grid_col]
end

# Function to convert VOC coordinates to Darknet coordinates
def convert_to_darknet(size, box)
  dw = 1.0 / size[0]
  dh = 1.0 / size[1]

  x = (box[0] + box[1]) / 2.0
  y = (box[2] + box[3]) / 2.0
  w = box[1] - box[0]
  h = box[3] - box[2]
  x = x * dw
  w = w * dw
  y = y * dh
  h = h * dh

  [x, y, w, h]
end

# Function to convert Darknet coordinates back to VOC coordinates
def convert_to_voc(size, darknet_coords)
  dw = size[0]
  dh = size[1]

  x = darknet_coords[0] * dw
  w = darknet_coords[2] * dw
  y = darknet_coords[1] * dh
  h = darknet_coords[3] * dh

  xmin = (x - w / 2.0).round
  xmax = (x + w / 2.0).round
  ymin = (y - h / 2.0).round
  ymax = (y + h / 2.0).round

  [xmin, xmax, ymin, ymax]
end

# Given image size and VOC coordinates
image_size = [500.0, 333.0]
voc_coords = [151, 349, 178, 236]
darknet_coords = [0.5000, 0.6216, 0.3960, 0.1742]

# Convert VOC to Darknet
darknet_result = convert_to_darknet(image_size, voc_coords)
printf("Darknet Coordinates: [%0.4f, %0.4f, %0.4f, %0.4f]\n", *darknet_result)

# Convert back from Darknet to VOC
voc_result = convert_to_voc(image_size, darknet_result)
printf("VOC Coordinates: [%d, %d, %d, %d]\n", *voc_result)

raise unless float_equal?(darknet_coords[0], darknet_result[0])
raise unless float_equal?(darknet_coords[1], darknet_result[1])
raise unless float_equal?(darknet_coords[2], darknet_result[2])
raise unless float_equal?(darknet_coords[3], darknet_result[3])
raise unless voc_result == voc_coords

grid_row, grid_col = darknet_to_grid(darknet_coords, GRID_SIZE)

printf("Darknet row: %d, col: %d\n", grid_row, grid_col)
