#!/bin/bash
# Copies predictions from container to local filesystem
# takes a single argument: the name of the division detection container

mkdir -p ~/data/division_detection/results
docker cp $1:/results/pretrained/data/dense ~/data/division_detection/results/
