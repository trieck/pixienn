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

void testModel()
{
    const auto model = Model::create("/home/trieck/work/pixienn/resources/models/yolov1-tiny.yml");
}

int main()
{
    try {
        testModel();
    } catch (const px::Error& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}
