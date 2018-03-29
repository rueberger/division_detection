#!/bin/bash
# takes a single argument: the path to the compressed data
# unzips compressed data and puts in ~/data/division_detection/projections  as expected
cd /tmp
mv $1 vols.zip
unzip vols.zip
mkdir -p ~/data/division_detection/klb
mkdir -p ~/data/division_detection/projections
mv Volume_*.projection.klb ~/data/division_detection/projections
mv Volume_*.klb ~/data/division_detection/klb
rm vols.zip
