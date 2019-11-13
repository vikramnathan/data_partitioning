import sys
import os
import argparse
import numpy as np
from progress.bar import ShadyBar
import multiprocessing as mp

def blocks_for_query(args):
    qfile = args[0]
    block_size = args[1]
    ids = np.fromfile(qfile, dtype=np.int32)
    blocks = np.unique(np.floor_divide(ids, block_size))
    return len(blocks)

def compute_cost(query_dir, block_size):
    cost = 0
    files = [os.path.join(query_dir, f) for f in os.listdir(query_dir)]
    args = zip(files, [block_size] * len(files))
    cpus = mp.cpu_count()
    bar = ShadyBar('Scanning Queries (%d cores)' % cpus, max = len(files))
    pool = mp.Pool(processes = cpus)
    for c in pool.imap_unordered(blocks_for_query, args):
        cost += c
        bar.next()
    bar.finish()
    print('Total blocks accessed: %d' % cost)


