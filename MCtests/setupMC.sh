#!/bin/bash

SCRATCHDIR=/scratch/ppmc    #needs to match in the submit script
MODELFILE=model
EPHEMFILE=ephemeris

mkdir -p ${SCRATCHDIR}
rsync -avuxP /home/tpennucc/PP/pplib_dist.py ${HOSTNAME}:${SCRATCHDIR}/
rsync -avuxP ${PBS_O_WORKDIR}/ppmc.py ${HOSTNAME}:${SCRATCHDIR}/
rsync -avuxP ${PBS_O_WORKDIR}/${MODELFILE} ${HOSTNAME}:${SCRATCHDIR}/
rsync -avuxP ${PBS_O_WORKDIR}/${EPHEMFILE} ${HOSTNAME}:${SCRATCHDIR}/
