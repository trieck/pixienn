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
#include <stdio.h>

int main(int argc, char** argv)
{
    cudaDeviceProp dP;
    float min_cc = 3.0;

    int rc = cudaGetDeviceProperties(&dP, 0);
    if (rc != cudaSuccess) {
        cudaError_t error = cudaGetLastError();
        printf("CUDA error: %s", cudaGetErrorString(error));
        return rc; /* Failure */
    }
    if ((dP.major + (dP.minor / 10)) < min_cc) {
        printf("Min Compute Capability of %2.1f required:  %d.%d found\n Not Building CUDA Code", min_cc, dP.major,
               dP.minor);
        return 1; /* Failure */
    } else {
        printf("-gencode arch=compute_%d%d,code=sm_%d%d", dP.major, dP.minor, dP.major, dP.minor);
        return 0; /* Success */
    }
}
