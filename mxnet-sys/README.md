# mxnet-sys

## Build

You should specify where `libmxnet.so` is. For example:

```bash
export MXNET_PATH=/home/sundoge/miniconda3/envs/mxnet1.3/lib/python3.6/site-packages/mxnet
export LD_LIBRARY_PATH=$MXNET_PATH:$LD_LIBRARY_PATH
cargo build
cargo test
```

## Status

There has been already crates [jakelee8/mxnet-sys
](https://github.com/jakelee8/mxnet-sys) and [jakelee8/mxnet-rs
](https://github.com/jakelee8/mxnet-rs) but they are not longer maintained.

This crate does not gurantee anything, but I will try my best.