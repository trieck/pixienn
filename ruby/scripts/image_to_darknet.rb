#!/bin/env ruby

require_relative '../utils/darknet'

GRID_SIZE = 7

# Function to compare floating-point numbers with a specified error threshold
def float_equal?(a, b, epsilon = 1e-4)
  (a - b).abs < epsilon
end


# Given image size and VOC coordinates
image_size = [500.0, 333.0]
img_coords = [151, 349, 178, 236]
darknet_coords = [0.5000, 0.6216, 0.3960, 0.1742]

# Convert Image coords to Darknet
darknet_result = image_to_darknet(image_size, img_coords)
printf("Darknet Coordinates: [%0.4f, %0.4f, %0.4f, %0.4f]\n", *darknet_result)

# Convert back from Darknet to Image coords
img_result = darknet_to_image(image_size, darknet_result)
printf("Image Coordinates: [%d, %d, %d, %d]\n", *img_result)

raise unless float_equal?(darknet_coords[0], darknet_result[0])
raise unless float_equal?(darknet_coords[1], darknet_result[1])
raise unless float_equal?(darknet_coords[2], darknet_result[2])
raise unless float_equal?(darknet_coords[3], darknet_result[3])
raise unless img_result == img_coords

grid_row, grid_col = darknet_to_grid(darknet_coords, GRID_SIZE)

printf("Darknet row: %d, col: %d\n", grid_row, grid_col)
