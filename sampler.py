import numpy as np
import os
import argparse
import sys
 
parser = argparse.ArgumentParser('Sampler')
parser.add_argument('--query-dir',
        required=True,
        help = 'Directory with binary query outputs')
parser.add_argument('--output-dir',
        required=True,
        help = 'Directory to output sampled query files')
parser.add_argument('--sampled-index-file',
        default='',
        type=str,
        help = 'File with random ids already stores')
parser.add_argument('--sample-ratio',
        type=float,
        default=0.001,
        help='The sampling ratio')
args = parser.parse_args()

assert (args.output_dir != args.query_dir)

MAX_RECORD_ID = 2000000000

def sample(sampled_id_dict, qfile):
    print('Sampling', qfile)
    ids = np.fromfile(os.path.join(args.query_dir, qfile), dtype=np.int32)
    print(len(ids))
    sampled_ids = set()
    for i in ids:
        if i in sampled_id_dict:
            sampled_ids.add(i)
    outfile = os.path.join(args.output_dir, qfile)
    id_lst = np.array(list(sampled_ids), dtype=np.int32)
    id_lst.sort()
    id_lst.tofile(os.path.join(args.output_dir, qfile))

def sample_dir():
    sample_size = int(args.sample_ratio * MAX_RECORD_ID)
    ids = None
    if os.path.exists(args.sampled_index_file):
        print('Found file with ids')
        ids = np.fromfile(args.sampled_index_file, dtype=np.int32)
    else:
        ids = np.random.choice(MAX_RECORD_ID,
            sample_size, replace=False).astype(dtype=np.int32)
        ids.sort()
        ids.tofile(args.sampled_index_file)
    id_dict = {}
    print('Finished generating IDs, length', len(ids))
    for i in ids:
        id_dict[i] = True
    print('Generated ID dictionary')
    if os.path.isdir(args.output_dir):
        print('Output directory already exists')
        sys.exit(1)
    os.mkdir(args.output_dir)
    for qfile in os.listdir(args.query_dir):
        sample(id_dict, qfile)

sample_dir()
