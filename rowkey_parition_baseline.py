import sys
import os
import argparse
import numpy as np
from progress.bar import ShadyBar
import multiprocessing as mp

## Implements a simple partition-by-rowkey baseline, where the row ID space is
# partitioned into blocks of the given size.

parser = argparse.ArgumentParser('Compute Intersections')
parser.add_argument('--query-dir',
        type=str,
        required=True,
        help='Directories with one file per query with the list of record IDs matching that query (binary)')
parser.add_argument('--block-size',
        type=int,
        required=True,
        help='Size of each block')
args = parser.parse_args()

def blocks_for_query(qfile):
    ids = np.fromfile(os.path.join(args.query_dir, qfile), dtype=np.int32)
    blocks = np.unique(np.floor_divide(ids, args.block_size))
    return len(blocks)

def compute_baseline():
    cost = 0
    files = os.listdir(args.query_dir)
    cpus = mp.cpu_count()
    bar = ShadyBar('Scanning Queries (%d cores)' % cpus, max = len(files))
    pool = mp.Pool(processes = cpus)
    for c in pool.imap_unordered(blocks_for_query, files):
        cost += c
        bar.next()
    bar.finish()
    print('Total blocks accessed: %d' % cost)

compute_baseline()


