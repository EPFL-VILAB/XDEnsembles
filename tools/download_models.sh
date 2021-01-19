##!/usr/bin/env bash

wget https://drive.switch.ch/index.php/s/6dFgYwR8dGj07jF/download
unzip download
mkdir -p models
mv xdomain-ensembles models
rm download