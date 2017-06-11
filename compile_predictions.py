
import pandas as pd
import numpy as np
from engine import Engine

engine = Engine()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('count_files', nargs='+')
parser.add_argument('-o', dest='output', required=True)
args = parser.parse_args()

counts_total = None

args.count_files = list(args.count_files)
args.count_files.reverse()

for path in args.count_files:
    print(path)
    counts = engine.load_counts(path)
    if counts_total is None:
        counts_total = counts
    else:
        counts_total.loc[counts.index] = counts
        # counts_total = counts_total.merge(counts, left_index=True, right_index=True)
    print(counts_total.shape)

counts_total = np.round(counts_total).astype(np.int32)
counts_total = counts_total.sort_index()

counts_total.to_csv(args.output)
