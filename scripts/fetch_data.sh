#!/bin/bash
# fetches data from dropbox and puts in ~/data/klb

wget https://www.dropbox.com/s/lvuoq22joc64lul/Volumes.zip?dl=0#
mv Volumes.zip\?dl\=0 vols.zip
unzip vols.zip
mkdir klb
mkdir -p ~/data/klb
mv Volume_*.klb ~/data/klb/
