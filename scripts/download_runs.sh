#!/bin/bash

# This script is meant to run on your local machine
# It will download the output files from the Palmetto cluster
# It will use one of the download nodes to xz compress the files before downloading them
# It will then decompress the files on your local machine
# Everything is seamless and this is essentially scp on steroids

mkdir -p ../analysis_many_dims
cd ../analysis_many_dims || exit
ssh USER@hpcdtn01.rcd.clemson.edu "cd /scratch/USER/KMC/analysis_many_dims && tar -cf - ./ | xz -T25 -c -9" | xz -d -c | tar -xf -