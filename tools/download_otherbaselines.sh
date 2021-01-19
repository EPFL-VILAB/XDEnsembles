##!/usr/bin/env bash

wget https://drive.switch.ch/index.php/s/ep2j3s8nC7QoqWV/download
unzip download
mkdir -p baselines
mv xdomain-ensembles-otherbaselines baselines
rm download