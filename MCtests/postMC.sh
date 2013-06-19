#!/bin/bash

SCRATCHDIR=/scratch/ppmc    #needs to match in the submit script

rsync -avuxP ${HOSTNAME}:${SCRATCHDIR}/*inf ${PBS_O_WORKDIR}
rsync -avuxP ${HOSTNAME}:${SCRATCHDIR}/*pick ${PBS_O_WORKDIR}
rm -rf %${SCRATCHDIR}
