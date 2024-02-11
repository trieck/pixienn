#!/bin/bash

path=$(pwd)/resources/data/coco/images

filename=train2017.zip
url=http://images.cocodataset.org/zips/${filename}

mkdir -p "${path}"
pushd "${path}" || exit

wget ${url}

unzip ${filename}
rm ${filename}

popd || exit




