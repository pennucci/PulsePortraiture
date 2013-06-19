#!/bin/bash

SCRATCHDIR=/scratch/ppmc

for i in 01 02 03 04 05 06 07 08 09 10
do
    rsync -avuxP nimrod$i:${SCRATCHDIR}/DM2.5e-03_5* .
done
