#!/bin/bash

path=$(pwd)/resources/data/voc

filename=VOCtrainval_06-Nov-2007.tar
#filename=VOCtest_06-Nov-2007.tar
#filename=VOCtrainval_11-May-2012.tar

url=https://pjreddie.com/media/files/${filename}

mkdir -p "${path}"
pushd "${path}" || exit

wget ${url}

tar xvf ${filename}
rm ${filename}

popd || exit
