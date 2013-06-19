#!/bin/bash

SCRATCHDIR=/scratch/ppmc/

rm -rf ${SCRATCHDIR}

for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19
do
    ssh nimrod$i 'rm -rf %s'%${SCRATCHDIR}
done
