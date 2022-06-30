#!/usr/bin/env python3
from train_VAE import get_ff_args, runFF

if __name__ == '__main__':
    config = get_ff_args()
    runFF(config)
