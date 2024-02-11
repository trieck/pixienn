require 'yaml'

module Config
  unless defined? CONFIG
    CONFIG_PATH = File.expand_path('config.yml', __dir__)
    CONFIG = YAML.load_file(CONFIG_PATH) if File.exist?(CONFIG_PATH)
  end

  PIXIENN_PATH = CONFIG&.fetch('pixienn_path', nil)
end
