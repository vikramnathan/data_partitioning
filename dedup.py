import numpy as np
import sys
import os
from progress.bar import ShadyBar
import argparse

# The purpose of this script is simply to deduplicate and resort the row IDs in each query
# file. There have been cases where they aren't unique, which screws up our
# computation.
parser = argparse.ArgumentParser('Deduplicate')
parser.add_argument('--source-dir',
        required=True,
        type=str,
        help='Directory with the original query files')
parser.add_argument('--output-dir',
        required=True,
        type=str,
        help='Directory to output to')
args = parser.parse_args()


existing_files = {}
if os.path.isdir(args.output_dir):
    for qfile in os.listdir(args.output_dir):
        existing_files[qfile] = True
else:
    os.mkdir(args.output_dir)

bar = ShadyBar('Progress', max=len(os.listdir(args.source_dir)))
for qfile in os.listdir(args.source_dir):
    bar.next()
    if qfile in existing_files:
        continue
    f = np.fromfile(os.path.join(args.source_dir, qfile), dtype=np.int32)
    f = np.unique(f)
    f.sort()
    f.tofile(os.path.join(args.output_dir, qfile))
bar.finish()

