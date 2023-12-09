module Constants
  PIXIENN_BUILD_PATH = File.expand_path('../../../cmake-build-debug', __FILE__)
  PIXIENN_BIN_PATH = "#{PIXIENN_BUILD_PATH}/bin"
  PIXIENN_PREDICTION_IMAGE = "#{PIXIENN_BUILD_PATH}/predictions.jpg"
  PIXIENN_PREDICTION_JSON = "#{PIXIENN_BUILD_PATH}/predictions.geojson"
end
