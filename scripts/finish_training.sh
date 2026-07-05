#!/bin/bash
python3 cli.py train --epochs 100 --seeds 42
python3 cli.py eval
