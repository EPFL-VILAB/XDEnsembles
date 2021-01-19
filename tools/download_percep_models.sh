##!/usr/bin/env bash

wget https://drive.switch.ch/index.php/s/aXu4EFaznqtNzsE/download
unzip download
rm download
mkdir -p models
mkdir -p models/perceps
mv percep_models/* models/perceps/
rmdir percep_models
