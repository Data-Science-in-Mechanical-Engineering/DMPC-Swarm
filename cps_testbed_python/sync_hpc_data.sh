#!/bin/bash

rsync -a -e ssh mf724021@copy18-1.hpc.itc.rwth-aachen.de:~/Dokumente/hpc_runs/ ../../hpc_runs/

# rsync -a -e ssh mf724021@copy18-1.hpc.itc.rwth-aachen.de:~/Dokumente/dampc_dataset/ ../../dampc_dataset/
