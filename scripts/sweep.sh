#!/bin/bash
for alpha in 0.1 0.5 1.0 2.0; do
  python3 cli.py train --cql_alpha $alpha
done
