#!/bin/bash

# rsync --progress -a -e ssh mf724021@copy23-1.hpc.itc.rwth-aachen.de:/work/p0022034/ ~/temp_datasets/AMPC/
rsync --progress -a -e ssh mf724021@copy23-1.hpc.itc.rwth-aachen.de:/work/mf724021/dmpc/ /data/hpc_runs/dmpc/
# rsync --progress -a -e ssh mf724021@copy23-1.hpc.itc.rwth-aachen.de:/work/mf724021/dmpc/ ../../hpc_runs/

# rsync -a -e ssh mf724021@copy18-1.hpc.itc.rwth-aachen.de:~/Dokumente/dampc_dataset/ ../../dampc_dataset/
