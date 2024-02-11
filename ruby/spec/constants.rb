require_relative '../load_config'

module Constants
  PIXIENN_PATH = Config::PIXIENN_PATH
  PIXIENN_BIN = "#{PIXIENN_PATH}/pixienn"
  PIXIENN_PREDICTION_IMAGE = "#{PIXIENN_PATH}/predictions.jpg"
  PIXIENN_PREDICTION_JSON = "#{PIXIENN_PATH}/predictions.geojson"
end
