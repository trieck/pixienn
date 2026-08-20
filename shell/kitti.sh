#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
data_dir=${root}/resources/data/kitti
download_dir=${data_dir}/.downloads
archive_dir=${download_dir}/extracted

mkdir -p "${download_dir}" "${archive_dir}"

image_archive=${download_dir}/data_object_image_2.zip
label_archive=${download_dir}/data_object_label_2.zip
image_url=https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
label_url=https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

if [[ ! -s ${image_archive} ]]; then
  wget -c -O "${image_archive}" "${image_url}"
fi
if [[ ! -s ${label_archive} ]]; then
  wget -c -O "${label_archive}" "${label_url}"
fi

if [[ ! -d ${archive_dir}/training/image_2 || ! -d ${archive_dir}/training/label_2 ]]; then
  unzip -q -o "${image_archive}" -d "${archive_dir}"
  unzip -q -o "${label_archive}" -d "${archive_dir}"
fi

python3 "${root}/tools/prepare_kitti.py" \
  --source "${archive_dir}" \
  --output "${data_dir}"

echo "KITTI Darknet data is ready under ${data_dir}"
