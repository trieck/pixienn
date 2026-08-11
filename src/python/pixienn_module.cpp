#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <variant>

#include "Model.h"
#include "PxTensor.h"

namespace py = pybind11;
namespace px {

class PythonTensor
{
public:
    explicit PythonTensor(const std::vector<std::size_t>& shape, float value = 0.0f)
    {
        switch (shape.size()) {
        case 1: tensor_ = std::make_shared<PxCpuTensor<1>>(asShape<1>(shape), value); break;
        case 2: tensor_ = std::make_shared<PxCpuTensor<2>>(asShape<2>(shape), value); break;
        case 3: tensor_ = std::make_shared<PxCpuTensor<3>>(asShape<3>(shape), value); break;
        case 4: tensor_ = std::make_shared<PxCpuTensor<4>>(asShape<4>(shape), value); break;
        default: throw std::invalid_argument("PxTensor binding supports ranks 1 through 4");
        }
    }

    PythonTensor(const std::vector<std::size_t>& shape, const std::vector<float>& values)
        : PythonTensor(shape)
    {
        if (values.size() != size()) {
            throw std::invalid_argument("tensor values must match tensor size");
        }
        std::visit([&values](auto& t) {
            for (std::size_t i = 0; i < values.size(); ++i) {
                (*t)[static_cast<int>(i)] = values[i];
            }
        }, tensor_);
    }

    std::vector<std::size_t> shape() const
    {
        return std::visit([](const auto& tensor) {
            const auto& shape = tensor->shape();
            return std::vector<std::size_t>(shape.begin(), shape.end());
        }, tensor_);
    }

    std::vector<std::size_t> strides() const
    {
        return std::visit([](const auto& tensor) {
            const auto& strides = tensor->strides();
            return std::vector<std::size_t>(strides.begin(), strides.end());
        }, tensor_);
    }

    std::size_t size() const { return std::visit([](const auto& t) { return t->size(); }, tensor_); }
    std::size_t dimSize(std::size_t dim) const
    {
        return std::visit([dim](const auto& t) {
            if (dim >= t->dims()) {
                throw py::index_error("tensor dimension out of range");
            }
            return t->size(dim);
        }, tensor_);
    }
    std::size_t ndim() const { return std::visit([](const auto& t) { return t->dims(); }, tensor_); }
    std::string device() const { return "cpu"; }
    std::vector<float> values() const { return std::visit([](const auto& t) { return t->asVector(); }, tensor_); }
    void fill(float value) { std::visit([value](auto& t) { t->fill(value); }, tensor_); }

    py::buffer_info buffer()
    {
        return std::visit([](auto& t) {
            std::vector<py::ssize_t> shape;
            std::vector<py::ssize_t> strides;
            for (const auto value : t->shape()) {
                shape.push_back(static_cast<py::ssize_t>(value));
            }
            for (const auto value : t->strides()) {
                strides.push_back(static_cast<py::ssize_t>(value * sizeof(float)));
            }
            return py::buffer_info(
                    t->data(), sizeof(float),
                    py::format_descriptor<float>::format(),
                    shape.size(), shape, strides);
        }, tensor_);
    }

    float at(py::ssize_t index) const
    {
        const auto normalized = normalizeIndex(index);
        return std::visit([normalized](const auto& t) {
            return (*t)[static_cast<int>(normalized)];
        }, tensor_);
    }

    void set(py::ssize_t index, float value)
    {
        const auto normalized = normalizeIndex(index);
        std::visit([normalized, value](auto& t) {
            (*t)[static_cast<int>(normalized)] = value;
        }, tensor_);
    }

    float atIndices(const py::tuple& indices) const
    {
        return at(flattenIndices(indices));
    }

    void setIndices(const py::tuple& indices, float value)
    {
        set(flattenIndices(indices), value);
    }

    PythonTensor add(float value) const { return scalarOp(value, [](float a, float b) { return a + b; }); }
    PythonTensor subtract(float value) const { return scalarOp(value, [](float a, float b) { return a - b; }); }
    PythonTensor multiply(float value) const { return scalarOp(value, [](float a, float b) { return a * b; }); }
    PythonTensor divide(float value) const
    {
        if (value == 0.0f) {
            throw py::value_error("division by zero");
        }
        return scalarOp(value, [](float a, float b) { return a / b; });
    }

    PythonTensor clone() const { return PythonTensor(shape(), values()); }

private:
    template<typename Operation>
    PythonTensor scalarOp(float value, Operation operation) const
    {
        auto result = PythonTensor(shape());
        const auto input = values();
        std::visit([&](auto& t) {
            for (std::size_t i = 0; i < input.size(); ++i) {
                (*t)[static_cast<int>(i)] = operation(input[i], value);
            }
        }, result.tensor_);
        return result;
    }

    std::size_t normalizeIndex(py::ssize_t index) const
    {
        const auto length = static_cast<py::ssize_t>(size());
        if (index < 0) {
            index += length;
        }
        if (index < 0 || index >= length) {
            throw py::index_error("tensor index out of range");
        }
        return static_cast<std::size_t>(index);
    }

    py::ssize_t flattenIndices(const py::tuple& indices) const
    {
        const auto dimensions = shape();
        const auto tensorStrides = strides();
        if (indices.size() != dimensions.size()) {
            throw py::index_error("number of indices must match tensor rank");
        }

        std::size_t offset = 0;
        for (py::ssize_t dimension = 0; dimension < indices.size(); ++dimension) {
            auto index = indices[dimension].cast<py::ssize_t>();
            const auto extent = static_cast<py::ssize_t>(dimensions[dimension]);
            if (index < 0) {
                index += extent;
            }
            if (index < 0 || index >= extent) {
                throw py::index_error("tensor index out of range");
            }
            offset += static_cast<std::size_t>(index) * tensorStrides[dimension];
        }
        return static_cast<py::ssize_t>(offset);
    }

    template<std::size_t N>
    static typename PxCpuTensor<N>::shape_type asShape(const std::vector<std::size_t>& shape)
    {
        typename PxCpuTensor<N>::shape_type result{};
        std::copy(shape.begin(), shape.end(), result.begin());
        return result;
    }

    std::variant<std::shared_ptr<PxCpuTensor<1>>, std::shared_ptr<PxCpuTensor<2>>,
                 std::shared_ptr<PxCpuTensor<3>>, std::shared_ptr<PxCpuTensor<4>>> tensor_;
};

class PythonModel
{
public:
    PythonModel(int channels, int height, int width, int batch)
        : model_(std::make_unique<px::Model<px::Device::CPU>>())
    {
        if (channels <= 0 || height <= 0 || width <= 0 || batch <= 0) {
            throw py::value_error("model dimensions and batch must be positive");
        }
        auto model = YAML::Node(YAML::NodeType::Map);
        model["channels"] = channels;
        model["height"] = height;
        model["width"] = width;
        model["batch"] = batch;
        model["layers"] = YAML::Node(YAML::NodeType::Sequence);
        document_["model"] = model;
    }

    PythonModel& addLayer(const py::dict& definition)
    {
        if (built_) {
            throw py::value_error("cannot add layers after model.build()");
        }
        document_["model"]["layers"].push_back(toYaml(definition));
        return *this;
    }

    PythonModel& setOptions(const py::dict& options)
    {
        if (built_) {
            throw py::value_error("model options must be set before model.build()");
        }
        for (const auto& item : options) {
            if (!py::isinstance<py::str>(item.first)) {
                throw py::type_error("model option keys must be strings");
            }
            const auto key = item.first.cast<std::string>();
            document_["model"][key] = toYaml(item.second);
            if (key == "weights-file") {
                model_->setWeightsFile(item.second.cast<std::string>());
            } else if (key == "backup-dir") {
                model_->setBackupDir(item.second.cast<std::string>());
            }
        }
        return *this;
    }

    void build()
    {
        if (!built_) {
            model_->parseModel(document_);
            built_ = true;
        }
    }

    PythonModel& configureTraining(const py::dict& definition)
    {
        if (built_) {
            throw py::value_error("training configuration must be set before model.build()");
        }
        document_["training"] = toYaml(definition);
        model_->setMode(Mode::TRAINING);
        return *this;
    }

    void train()
    {
        if (!built_) {
            throw py::value_error("model must be built before train()");
        }
        model_->train();
    }

    void evaluate()
    {
        if (!built_) {
            throw py::value_error("model must be built before evaluate()");
        }
        model_->evaluate();
    }

    void loadWeights(const std::string& fileName)
    {
        if (!built_) {
            throw py::value_error("model must be built before loading weights");
        }
        model_->loadWeightsFile(fileName);
    }

    void saveWeights(const std::string& fileName)
    {
        if (!built_) {
            throw py::value_error("model must be built before saving weights");
        }
        model_->saveWeightsFile(fileName);
    }

    void saveTrainingState(const std::string& fileName)
    {
        if (!built_) {
            throw py::value_error("model must be built before saving training state");
        }
        model_->saveTrainingStateFile(fileName);
    }

    void setMode(const std::string& mode)
    {
        if (mode == "inference") model_->setMode(Mode::INFERRING);
        else if (mode == "training") model_->setMode(Mode::TRAINING);
        else if (mode == "validation") model_->setMode(Mode::VALIDATING);
        else throw py::value_error("mode must be inference, training, or validation");
    }

    std::string mode() const
    {
        if (model_->training()) return "training";
        if (model_->validating()) return "validation";
        return "inference";
    }

    std::string predictJson(const std::string& imageFile, float nmsThreshold = 0.3f)
    {
        if (!built_) {
            throw py::value_error("model must be built before predict()");
        }
        auto detects = model_->predict(imageFile);
        return model_->asJson(nms(detects, nmsThreshold));
    }

    std::string predictImage(const std::string& imageFile, float nmsThreshold = 0.3f)
    {
        if (!built_) {
            throw py::value_error("model must be built before predict()");
        }
        auto detects = nms(model_->predict(imageFile), nmsThreshold);
        model_->overlay(imageFile, detects);
        return model_->asJson(detects);
    }

    void setLabels(const std::vector<std::string>& labels) { model_->setLabels(labels); }
    std::vector<std::string> labels() const { return model_->labels(); }
    void setThreshold(float threshold) { model_->setThreshold(threshold); }
    float threshold() const noexcept { return model_->threshold(); }
    int classes() const noexcept { return model_->classes(); }
    int channels() const noexcept { return model_->channels(); }
    int height() const noexcept { return model_->height(); }
    int width() const noexcept { return model_->width(); }
    float momentum() const noexcept { return model_->momentum(); }
    float decay() const noexcept { return model_->decay(); }
    bool adamEnabled() const noexcept { return model_->adamEnabled(); }
    float adamBeta1() const noexcept { return model_->adamBeta1(); }
    float adamBeta2() const noexcept { return model_->adamBeta2(); }
    float adamEpsilon() const noexcept { return model_->adamEpsilon(); }
    float cost() const noexcept { return model_->cost(); }
    float learningRate() const { return model_->learningRate(); }
    std::size_t seen() const noexcept { return model_->seen(); }
    int batch() const noexcept { return model_->batch(); }
    std::vector<std::vector<int>> layerShapes() const
    {
        std::vector<std::vector<int>> result;
        result.reserve(model_->layerSize());
        for (int i = 0; i < model_->layerSize(); ++i) {
            const auto& layer = model_->layerAt(i);
            result.push_back({layer->outChannels(), layer->outHeight(), layer->outWidth()});
        }
        return result;
    }

    PythonTensor forward(const PythonTensor& input)
    {
        if (!built_) {
            throw py::value_error("model must be built before forward()");
        }
        const auto expected = static_cast<std::size_t>(model_->batch()) *
                              static_cast<std::size_t>(model_->channels()) *
                              static_cast<std::size_t>(model_->height()) *
                              static_cast<std::size_t>(model_->width());
        if (input.size() != expected) {
            throw py::value_error("input tensor size does not match model input");
        }
        const auto inputValues = input.values();
        PxCpuVector values(inputValues.size());
        values.copyHost(inputValues.data(), inputValues.size());
        model_->forward(values);
        const auto& last = model_->layerAt(model_->layerSize() - 1);
        const auto& output = last->output();
        const auto outputShape = std::vector<std::size_t>{
            static_cast<std::size_t>(model_->batch()),
            static_cast<std::size_t>(last->outChannels()),
            static_cast<std::size_t>(last->outHeight()),
            static_cast<std::size_t>(last->outWidth())};
        return PythonTensor(outputShape, output.asVector());
    }

    bool built() const noexcept { return built_; }
    int layerCount() const noexcept { return model_->layerSize(); }
    std::vector<int> inputShape() const
    {
        return {model_->channels(), model_->height(), model_->width()};
    }

    std::vector<int> outputShape() const
    {
        if (!built_ || model_->layerSize() == 0) {
            return inputShape();
        }
        const auto& layer = model_->layerAt(model_->layerSize() - 1);
        return {layer->outChannels(), layer->outHeight(), layer->outWidth()};
    }

public:
    static YAML::Node toYaml(const py::handle& value)
    {
        if (value.is_none()) {
            return YAML::Node();
        }
        if (py::isinstance<py::dict>(value)) {
            auto result = YAML::Node(YAML::NodeType::Map);
            for (const auto& item : value.cast<py::dict>()) {
                if (!py::isinstance<py::str>(item.first)) {
                    throw py::type_error("model definition keys must be strings");
                }
                result[item.first.cast<std::string>()] = toYaml(item.second);
            }
            return result;
        }
        if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
            auto result = YAML::Node(YAML::NodeType::Sequence);
            for (const auto& item : value) {
                result.push_back(toYaml(item));
            }
            return result;
        }
        if (py::isinstance<py::bool_>(value)) {
            return YAML::Node(value.cast<bool>());
        }
        if (py::isinstance<py::int_>(value)) {
            return YAML::Node(value.cast<long long>());
        }
        if (py::isinstance<py::float_>(value)) {
            return YAML::Node(value.cast<double>());
        }
        if (py::isinstance<py::str>(value)) {
            return YAML::Node(value.cast<std::string>());
        }
        throw py::type_error("unsupported Python value in model definition");
    }

private:
    std::unique_ptr<px::Model<px::Device::CPU>> model_;
    YAML::Node document_ = YAML::Node(YAML::NodeType::Map);
    bool built_ = false;
};

#ifdef USE_CUDA
class PythonCudaTensor
{
public:
    explicit PythonCudaTensor(const std::vector<std::size_t>& shape, float value = 0.0f)
    {
        switch (shape.size()) {
        case 1: tensor_ = std::make_shared<px::PxCudaTensor<1>>(asShape<1>(shape), value); break;
        case 2: tensor_ = std::make_shared<px::PxCudaTensor<2>>(asShape<2>(shape), value); break;
        case 3: tensor_ = std::make_shared<px::PxCudaTensor<3>>(asShape<3>(shape), value); break;
        case 4: tensor_ = std::make_shared<px::PxCudaTensor<4>>(asShape<4>(shape), value); break;
        default: throw std::invalid_argument("CUDA tensor binding supports ranks 1 through 4");
        }
    }

    PythonCudaTensor(const std::vector<std::size_t>& shape, const std::vector<float>& values)
        : PythonCudaTensor(shape)
    {
        if (values.size() != size()) throw std::invalid_argument("tensor values must match tensor size");
        px::PxCpuVector host(values.size());
        host.copyHost(values.data(), values.size());
        std::visit([&host](auto& tensor) { tensor = std::make_shared<std::decay_t<decltype(*tensor)>>(tensor->shape(), host); }, tensor_);
    }

    std::vector<std::size_t> shape() const { return std::visit([](const auto& t) { const auto& s=t->shape(); return std::vector<std::size_t>(s.begin(),s.end()); }, tensor_); }
    std::vector<std::size_t> strides() const { return std::visit([](const auto& t) { const auto& s=t->strides(); return std::vector<std::size_t>(s.begin(),s.end()); }, tensor_); }
    std::size_t size() const { return std::visit([](const auto& t) { return t->size(); }, tensor_); }
    std::size_t ndim() const { return std::visit([](const auto& t) { return t->dims(); }, tensor_); }
    std::string device() const { return "cuda"; }
    std::vector<float> values() const { return std::visit([](const auto& t) { return t->asVector(); }, tensor_); }
    void fill(float value) { std::visit([value](auto& t) { t->fill(value); }, tensor_); }
    PythonCudaTensor clone() const { return PythonCudaTensor(shape(), values()); }
    float at(py::ssize_t index) const { auto v=values(); if (index<0) index += static_cast<py::ssize_t>(v.size()); if (index<0 || index>=static_cast<py::ssize_t>(v.size())) throw py::index_error("tensor index out of range"); return v[static_cast<std::size_t>(index)]; }

private:
    template<std::size_t N> static typename px::PxCudaTensor<N>::shape_type asShape(const std::vector<std::size_t>& shape) { typename px::PxCudaTensor<N>::shape_type result{}; std::copy(shape.begin(), shape.end(), result.begin()); return result; }
    std::variant<std::shared_ptr<px::PxCudaTensor<1>>, std::shared_ptr<px::PxCudaTensor<2>>, std::shared_ptr<px::PxCudaTensor<3>>, std::shared_ptr<px::PxCudaTensor<4>>> tensor_;
};

class PythonCudaModel
{
public:
    PythonCudaModel(int channels, int height, int width, int batch)
        : model_(std::make_unique<px::CudaModel>())
    {
        if (channels <= 0 || height <= 0 || width <= 0 || batch <= 0) {
            throw py::value_error("model dimensions and batch must be positive");
        }
        auto model = YAML::Node(YAML::NodeType::Map);
        model["channels"] = channels;
        model["height"] = height;
        model["width"] = width;
        model["batch"] = batch;
        model["layers"] = YAML::Node(YAML::NodeType::Sequence);
        document_["model"] = model;
    }

    PythonCudaModel& setOptions(const py::dict& options)
    {
        if (built_) throw py::value_error("model options must be set before build()");
        for (const auto& item : options) {
            if (!py::isinstance<py::str>(item.first)) throw py::type_error("model option keys must be strings");
            document_["model"][item.first.cast<std::string>()] = PythonModel::toYaml(item.second);
        }
        return *this;
    }

    PythonCudaModel& addLayer(const py::dict& definition)
    {
        if (built_) throw py::value_error("cannot add layers after model.build()");
        document_["model"]["layers"].push_back(PythonModel::toYaml(definition));
        return *this;
    }

    PythonCudaModel& setLabels(const std::vector<std::string>& labels)
    {
        model_->setLabels(labels);
        return *this;
    }

    void build()
    {
        if (!built_) {
            model_->parseModel(document_);
            built_ = true;
        }
    }

    PythonTensor forward(const PythonTensor& input)
    {
        if (!built_) throw py::value_error("model must be built before forward()");
        const auto expected = static_cast<std::size_t>(model_->batch()) * model_->channels() *
                              model_->height() * model_->width();
        if (input.size() != expected) throw py::value_error("input tensor size does not match model input");
        const auto values = input.values();
        px::PxCudaVector deviceInput(values.data(), values.data() + values.size());
        model_->forward(deviceInput);
        const auto& last = model_->layerAt(model_->layerSize() - 1);
        return PythonTensor({static_cast<std::size_t>(model_->batch()),
                             static_cast<std::size_t>(last->outChannels()),
                             static_cast<std::size_t>(last->outHeight()),
                             static_cast<std::size_t>(last->outWidth())}, last->output().asVector());
    }

    PythonCudaTensor forward(const PythonCudaTensor& input)
    {
        if (!built_) throw py::value_error("model must be built before forward()");
        const auto expected = static_cast<std::size_t>(model_->batch()) * model_->channels() * model_->height() * model_->width();
        if (input.size() != expected) throw py::value_error("input tensor size does not match model input");
        const auto values = input.values();
        px::PxCudaVector deviceInput(values.data(), values.data() + values.size());
        model_->forward(deviceInput);
        const auto& last = model_->layerAt(model_->layerSize() - 1);
        return PythonCudaTensor({static_cast<std::size_t>(model_->batch()), static_cast<std::size_t>(last->outChannels()),
                                 static_cast<std::size_t>(last->outHeight()), static_cast<std::size_t>(last->outWidth())},
                                last->output().asVector());
    }

    void loadWeights(const std::string& fileName) { model_->loadWeightsFile(fileName); }
    PythonCudaModel& setThreshold(float threshold) { model_->setThreshold(threshold); return *this; }
    std::string predictImage(const std::string& imageFile, float nmsThreshold = 0.3f)
    {
        if (!built_) throw py::value_error("model must be built before predict_image()");
        auto detects = nms(model_->predict(imageFile), nmsThreshold);
        model_->overlay(imageFile, detects);
        return model_->asJson(detects);
    }
    std::vector<int> outputShape() const
    {
        if (!built_ || model_->layerSize() == 0) return {model_->channels(), model_->height(), model_->width()};
        const auto& layer = model_->layerAt(model_->layerSize() - 1);
        return {layer->outChannels(), layer->outHeight(), layer->outWidth()};
    }
    bool built() const noexcept { return built_; }
    std::string device() const { return "cuda"; }
    int channels() const noexcept { return model_->channels(); }
    int height() const noexcept { return model_->height(); }
    int width() const noexcept { return model_->width(); }
    int batch() const noexcept { return model_->batch(); }

private:
    std::unique_ptr<px::CudaModel> model_;
    YAML::Node document_ = YAML::Node(YAML::NodeType::Map);
    bool built_ = false;
};
#endif

} // namespace px

PYBIND11_MODULE(_native, module)
{
    module.doc() = "Native PixieNN bindings";
    py::class_<px::PythonTensor>(module, "Tensor", py::buffer_protocol())
        .def(py::init<const std::vector<std::size_t>&, float>(),
             py::arg("shape"), py::arg("fill_value") = 0.0f)
        .def(py::init<const std::vector<std::size_t>&, const std::vector<float>&>(),
             py::arg("shape"), py::arg("values"))
        .def_property_readonly("shape", &px::PythonTensor::shape)
        .def_property_readonly("strides", &px::PythonTensor::strides)
        .def_property_readonly("size", &px::PythonTensor::size)
        .def("dim_size", &px::PythonTensor::dimSize, py::arg("dim"))
        .def_property_readonly("ndim", &px::PythonTensor::ndim)
        .def_property_readonly("device", &px::PythonTensor::device)
        .def("values", &px::PythonTensor::values)
        .def("fill", &px::PythonTensor::fill, py::arg("value"))
        .def_buffer(&px::PythonTensor::buffer)
        .def("__len__", &px::PythonTensor::size)
        .def("__getitem__", [](const px::PythonTensor& tensor, const py::object& index) {
            if (py::isinstance<py::tuple>(index)) {
                return tensor.atIndices(index.cast<py::tuple>());
            }
            return tensor.at(index.cast<py::ssize_t>());
        })
        .def("__setitem__", [](px::PythonTensor& tensor, const py::object& index, float value) {
            if (py::isinstance<py::tuple>(index)) {
                tensor.setIndices(index.cast<py::tuple>(), value);
            } else {
                tensor.set(index.cast<py::ssize_t>(), value);
            }
        })
        .def("__add__", &px::PythonTensor::add)
        .def("__sub__", &px::PythonTensor::subtract)
        .def("__mul__", &px::PythonTensor::multiply)
        .def("__truediv__", &px::PythonTensor::divide)
        .def("clone", &px::PythonTensor::clone);

    py::class_<px::PythonModel>(module, "Model")
        .def(py::init<int, int, int, int>(),
             py::arg("channels"), py::arg("height"), py::arg("width"),
             py::arg("batch") = 1)
        .def("add_layer", &px::PythonModel::addLayer,
             py::arg("definition"), py::return_value_policy::reference_internal)
        .def("set_options", &px::PythonModel::setOptions,
             py::arg("options"), py::return_value_policy::reference_internal)
        .def("build", &px::PythonModel::build)
        .def("configure_training", &px::PythonModel::configureTraining,
             py::arg("definition"), py::return_value_policy::reference_internal)
        .def("train", &px::PythonModel::train)
        .def("evaluate", &px::PythonModel::evaluate)
        .def("load_weights", &px::PythonModel::loadWeights, py::arg("file_name"))
        .def("save_weights", &px::PythonModel::saveWeights, py::arg("file_name"))
        .def("save_training_state", &px::PythonModel::saveTrainingState, py::arg("file_name"))
        .def("set_mode", &px::PythonModel::setMode, py::arg("mode"))
        .def_property_readonly("mode", &px::PythonModel::mode)
        .def("predict_json", &px::PythonModel::predictJson,
             py::arg("image_file"), py::arg("nms_threshold") = 0.3f)
        .def("predict_image", &px::PythonModel::predictImage,
             py::arg("image_file"), py::arg("nms_threshold") = 0.3f)
        .def("set_labels", &px::PythonModel::setLabels, py::arg("labels"))
        .def_property_readonly("labels", &px::PythonModel::labels)
        .def("set_threshold", &px::PythonModel::setThreshold, py::arg("threshold"))
        .def_property_readonly("threshold", &px::PythonModel::threshold)
        .def_property_readonly("classes", &px::PythonModel::classes)
        .def_property_readonly("channels", &px::PythonModel::channels)
        .def_property_readonly("height", &px::PythonModel::height)
        .def_property_readonly("width", &px::PythonModel::width)
        .def_property_readonly("momentum", &px::PythonModel::momentum)
        .def_property_readonly("decay", &px::PythonModel::decay)
        .def_property_readonly("adam_enabled", &px::PythonModel::adamEnabled)
        .def_property_readonly("adam_beta1", &px::PythonModel::adamBeta1)
        .def_property_readonly("adam_beta2", &px::PythonModel::adamBeta2)
        .def_property_readonly("adam_epsilon", &px::PythonModel::adamEpsilon)
        .def_property_readonly("cost", &px::PythonModel::cost)
        .def_property_readonly("learning_rate", &px::PythonModel::learningRate)
        .def_property_readonly("seen", &px::PythonModel::seen)
        .def_property_readonly("batch", &px::PythonModel::batch)
        .def_property_readonly("layer_shapes", &px::PythonModel::layerShapes)
        .def("forward", &px::PythonModel::forward, py::arg("input"))
        .def_property_readonly("built", &px::PythonModel::built)
        .def_property_readonly("layer_count", &px::PythonModel::layerCount)
        .def_property_readonly("input_shape", &px::PythonModel::inputShape)
        .def_property_readonly("output_shape", &px::PythonModel::outputShape);
#ifdef USE_CUDA
    py::class_<px::PythonCudaTensor>(module, "CudaTensor")
        .def(py::init<const std::vector<std::size_t>&, float>(), py::arg("shape"), py::arg("fill_value") = 0.0f)
        .def(py::init<const std::vector<std::size_t>&, const std::vector<float>&>(), py::arg("shape"), py::arg("values"))
        .def_property_readonly("shape", &px::PythonCudaTensor::shape)
        .def_property_readonly("strides", &px::PythonCudaTensor::strides)
        .def_property_readonly("size", &px::PythonCudaTensor::size)
        .def_property_readonly("ndim", &px::PythonCudaTensor::ndim)
        .def_property_readonly("device", &px::PythonCudaTensor::device)
        .def("values", &px::PythonCudaTensor::values)
        .def("fill", &px::PythonCudaTensor::fill, py::arg("value"))
        .def("clone", &px::PythonCudaTensor::clone)
        .def("__len__", &px::PythonCudaTensor::size)
        .def("__getitem__", &px::PythonCudaTensor::at);
    py::class_<px::PythonCudaModel>(module, "CudaModel")
        .def(py::init<int, int, int, int>(), py::arg("channels"), py::arg("height"),
             py::arg("width"), py::arg("batch") = 1)
        .def("set_options", &px::PythonCudaModel::setOptions, py::arg("options"),
             py::return_value_policy::reference_internal)
        .def("add_layer", &px::PythonCudaModel::addLayer, py::arg("definition"),
             py::return_value_policy::reference_internal)
        .def("set_labels", &px::PythonCudaModel::setLabels, py::arg("labels"),
             py::return_value_policy::reference_internal)
        .def("build", &px::PythonCudaModel::build)
        .def("forward", py::overload_cast<const px::PythonTensor&>(&px::PythonCudaModel::forward), py::arg("input"))
        .def("forward", py::overload_cast<const px::PythonCudaTensor&>(&px::PythonCudaModel::forward), py::arg("input"))
        .def("load_weights", &px::PythonCudaModel::loadWeights, py::arg("file_name"))
        .def("set_threshold", &px::PythonCudaModel::setThreshold, py::arg("threshold"),
             py::return_value_policy::reference_internal)
        .def("predict_image", &px::PythonCudaModel::predictImage,
             py::arg("image_file"), py::arg("nms_threshold") = 0.3f)
        .def_property_readonly("built", &px::PythonCudaModel::built)
        .def_property_readonly("device", &px::PythonCudaModel::device)
        .def_property_readonly("output_shape", &px::PythonCudaModel::outputShape);
#endif
}
