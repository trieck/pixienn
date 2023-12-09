require 'fileutils'
require 'json'
require 'rspec'

require_relative 'constants'
include Constants

describe 'Pixienn Tiny Yolov1 Inference Test' do
  before(:all) do
    @inference_command = "bin/pixienn " \
      "--no-gpu " \
      "--color-map=plasma " \
      "--nms=0.3 " \
      "--confidence=0.2 " \
      "../resources/cfg/yolov1-tiny-cfg.yml " \
      "../resources/images/dog.jpg"
  end

  after(:all) do
    FileUtils.rm("#{PIXIENN_PREDICTION_IMAGE}") if File.exist?("#{PIXIENN_PREDICTION_IMAGE}")
    FileUtils.rm("#{PIXIENN_PREDICTION_JSON}") if File.exist?("#{PIXIENN_PREDICTION_JSON}")
  end

  it 'runs Tiny Yolov1 model and checks generated files' do
    Dir.chdir(PIXIENN_BUILD_PATH) do
      `#{@inference_command}`
    end

    expect($?.success?).to be(true)

    expect(File.exist?("#{PIXIENN_PREDICTION_IMAGE}")).to be(true)
    expect(`file "#{PIXIENN_PREDICTION_IMAGE}"`).to include('JPEG image data')

    json = File.read("#{PIXIENN_PREDICTION_JSON}")
    predictions = JSON.parse(json)

    expect(predictions['features']).to be_a(Array)

    expected_predictions = [
      { 'class' => 'dog', 'confidence' => 0.4 },
      { 'class' => 'bicycle', 'confidence' => 0.25 },
      { 'class' => 'car', 'confidence' => 0.3 }
    ]

    predictions['features'].each do |feature|
      properties = feature['properties']
      expected = expected_predictions.find { |e| e['class'] == properties['class'] }

      expect(expected).not_to be_nil

      # Check the confidence within tolerance
      expect(properties['confidence']).to be_within(0.05).of(expected['confidence'])
    end

  end
end
