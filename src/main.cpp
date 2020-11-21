/********************************************************************************
* Copyright 2020 Thomas A. Rieck, All Rights Reserved
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

#include <iostream>

#include "Error.h"
#include "Model.h"
#include "Image.h"

#include "xtensor/xio.hpp"

using namespace px;

void showLayers(const cv::Mat& image, const char* baseName)
{
    int i;
    char buff[256];
    for (i = 0; i < image.channels(); ++i) {
        sprintf(buff, "%s - Layer %d", baseName, i);
        auto layer = px::imchannel(image, i);

        cv::namedWindow(buff, cv::WINDOW_AUTOSIZE);
        cv::imshow(buff, layer);
    }
}

void testConvolve()
{
    auto image = px::imread("resources/images/dog.jpg");
    auto kernel = px::imrandom(2, 2, image.channels());
    auto edge = px::immake(image.rows, image.cols, 1);

    px::imconvolve(image, kernel, 1, 0, edge);

    showLayers(edge, "Test Convolve");

    cv::waitKey();
}

void testYolo()
{
    auto model = Model("resources/models/yolov1-tiny.yml");

    std::cout << "Loading weights...";
    model.loadDarknetWeights("resources/weights/yolov1-tiny.weights");
    std::cout << "done." << std::endl;

    auto image = px::imread("resources/images/dog.jpg");
    auto sized = px::imletterbox(image, model.width(), model.height());
    auto input = px::imarray(sized);

    std::cout << "Running network...";

    auto detects = model.predict(std::move(input), image.cols, image.rows, 0.09f);

    for (const auto& det: detects) {
        for (auto i = 0; i < det.size(); ++i) {
            if (det[i] >= 0.09f) {
                printf("%.0f%%\n", det[i] * 100);
            }
        }
    }

    std::cout << "done." << std::endl;
}

int main()
{
    try {
        testYolo();
    } catch (const px::Error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}
