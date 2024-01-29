# Function to determine grid row and column for Darknet coordinates
def darknet_to_grid(darknet_coords, grid_size)
  x, y, _, _ = darknet_coords

  # Calculate grid row and column
  grid_row = (y * grid_size).to_i
  grid_col = (x * grid_size).to_i

  [grid_row, grid_col]
end

# Function to convert Image coordinates to Darknet coordinates
def image_to_darknet(size, box)
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

# Function to convert Darknet coordinates to Image coordinates
def darknet_to_image(size, darknet_coords)
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
