/********************************************************************************
* Copyright 2023 Thomas A. Rieck, All Rights Reserved
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
********************************************************************************/

#include <boost/endian/conversion.hpp>

#include <opencv2/core/utility.hpp>
#include <tiffio.h>
#include "Common.h"
#include "Error.h"
#include "TiffIO.h"

namespace px {

class TIFFReader
{
public:
    TIFFReader(const char* path);
    ~TIFFReader();

    uint32_t field(uint32_t tag) const;
    cv::Mat read();

private:
    using buffer_ptr = std::unique_ptr<uint8_t[]>;
    buffer_ptr scanLineBuffer() const;
    float maxValue() const;

    TIFF* tiff_ = nullptr;
};

TIFFReader::TIFFReader(const char* path)
{
    tiff_ = TIFFOpen(path, "r");
    PX_CHECK(tiff_, "Could not open image \"%s\".", path);
}

TIFFReader::~TIFFReader()
{
    if (tiff_ != nullptr) {
        TIFFClose(tiff_);
        tiff_ = nullptr;
    }
}

uint32_t TIFFReader::field(uint32_t tag) const
{
    uint32_t value = 0;
    TIFFGetField(tiff_, tag, &value);
    return value;
}

TIFFReader::buffer_ptr TIFFReader::scanLineBuffer() const
{
    auto scanLineSize = TIFFScanlineSize(tiff_);

    auto results = std::make_unique<uint8_t[]>(scanLineSize);

    return results;
}

float TIFFReader::maxValue() const
{
    auto bitsPerSample = field(TIFFTAG_BITSPERSAMPLE);
    auto sampleFormat = field(TIFFTAG_SAMPLEFORMAT);

    switch (sampleFormat) {
    case 0:
    case SAMPLEFORMAT_VOID:
    case SAMPLEFORMAT_INT:
    case SAMPLEFORMAT_UINT:
        return static_cast<float>((1ULL << bitsPerSample) - 1);
    case SAMPLEFORMAT_IEEEFP:
        if (bitsPerSample == 16 || bitsPerSample == 32) {
            return std::numeric_limits<float>::max();
        } else if (bitsPerSample == 64) {
            return static_cast<float>(std::numeric_limits<double>::max());
        }
    }

    PX_ERROR_THROW("Unsupported data format.");

    return 0;
}

cv::Mat TIFFReader::read()
{
    auto width = field(TIFFTAG_IMAGEWIDTH);
    auto height = field(TIFFTAG_IMAGELENGTH);
    auto numChannels = field(TIFFTAG_SAMPLESPERPIXEL);
    auto bitsPerSample = field(TIFFTAG_BITSPERSAMPLE);
    auto sampleFormat = field(TIFFTAG_SAMPLEFORMAT);

    cv::Mat image(height, width, CV_MAKETYPE(CV_32F, numChannels));

    auto max = maxValue();
    auto buffer = scanLineBuffer();
    auto elementSize = bitsPerSample / 8;

    for (auto row = 0; row < height; row++) {
        TIFFReadScanline(tiff_, buffer.get(), row);
        auto rowPtr = image.ptr<float>(row);

        for (auto col = 0; col < width; col++) {
            auto index = col * numChannels;

            for (auto channel = 0; channel < numChannels; channel++) {
                auto outIndex = index + channel;
                auto inIndex = outIndex * elementSize;

                float value = 0;
                if (bitsPerSample == 8) {
                    value = static_cast<float>(buffer[inIndex]);
                } else if (bitsPerSample == 16) {
                    value = *reinterpret_cast<uint16_t*>(&buffer[inIndex]);
                } else if (bitsPerSample == 32) {
                    if (sampleFormat == SAMPLEFORMAT_IEEEFP) {
                        value = *reinterpret_cast<float*>(&buffer[inIndex]);
                    } else {
                        value = *reinterpret_cast<uint32_t*>(&buffer[inIndex]);
                    }
                } else {
                    PX_ERROR_THROW("Unsupported image format bps=%d, sampleformat=%d.",
                                   bitsPerSample, sampleFormat);
                }

                rowPtr[outIndex] = (bitsPerSample == 32 && sampleFormat == SAMPLEFORMAT_IEEEFP) ? value : value / max;
            }
        }
    }

    return image;
}

cv::Mat readTIFF(const char* path)
{
    TIFFReader reader(path);
    return reader.read();
}

}   // px
