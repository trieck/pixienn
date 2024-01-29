class Darknet
  attr_accessor :filename
  attr_accessor :cfg

  include Enumerable

  def initialize(filename)
    @filename = filename
    @cfg = []
    read
  end

  def self.load_config(filename)
    return unless File.file? filename
    new(filename)
  end

  def read
    File.open(@filename, "r") {
      |fd| parse fd
    }
  end

  def parse(content)
    parser = Parser.new
    @cfg = parser.parse(content)
  end

  def each
    return unless block_given?
    i = 0
    @cfg.each do |section|
      yield section, i
      i += 1
    end
    self
  end
end

class Section
  include Enumerable

  attr_accessor :name
  attr_accessor :props

  def initialize(name)
    @name = name
    @props = {}
  end

  def add_property(key, value)
    @props[key] = value
  end

  def each
    return unless block_given?
    @props.each do |key, value|
      yield key, value
    end
    self
  end
end

class Parser

  def initialize
    comment = "\\s*(?:[#].*)?\\z"
    @section_regexp = %r/\A\s*\[([^\]]+)\]#{comment}/
    @property_regexp = %r/\A([^#]*?)(?<!\\)=(.*)\z/
  end

  def parse(content)
    sections = []
    curr_section = nil

    content.each_line do |line|
      @line = line.chomp

      case @line
      when @section_regexp
        curr_section = Section.new $1
        sections.append(curr_section)
      when @property_regexp
        raise "No section found for #{@line}" unless curr_section
        property = $1.strip
        value = parse_value $2.strip
        curr_section.add_property(property, value)
      else
        # skip
      end
    end

    sections
  end

  def to_number(value)
    Integer(value, exception: false) || Float(value, exception: false)
  end

  def parse_list(value)
    a = value.split(',')
    if a.length == 1
      a = to_number(value)
      a = value if a.nil?
      a
    else
      b = []
      a.each { |v|
        x = parse_value v
        b.append(x)
      }
      b
    end
  end

  def parse_value(value)
    a = to_number(value)
    if a.nil? # not a number
      a = parse_list(value)
    end
    a
  end
end
