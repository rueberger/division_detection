#!/bin/bash
# fetches data from dropbox and puts in ~/data/divison_detection/projections

cd /tmp
wget https://www.dropbox.com/s/lvuoq22joc64lul/Volumes.zip?dl=0#
mv Volumes.zip\?dl\=0 vols.zip
unzip vols.zip
mkdir -p ~/data/division_detection/klb
mkdir -p ~/data/division_detection/projections
mv Volume_*.projection.klb ~/data/division_detection/projections
mv Volume_*.klb ~/data/division_detection/klb
