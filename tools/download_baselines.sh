##!/usr/bin/env bash

wget https://drive.switch.ch/index.php/s/WVH91pZfgG2VnhB/download
unzip download
mkdir -p models
mv xdomain-ensembles-baselines/* models/
rm download
rm -r xdomain-ensembles-baselines
