## When local cuda missmatches pytorch12.2:

export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}


## (better pip install -U bitsandsytes) (when Error invalid device ordinal at line 388 in file /mmfs1/gscratch/zlab/timdettmers/git/bitsandbytes/csrc/pythonInterface.c):

bitsandbytes 0.40.0 works with cuda12.2
