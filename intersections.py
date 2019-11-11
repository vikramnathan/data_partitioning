import numpy as np
import shutil
import sys
import os
from array import array
from progress.bar import ShadyBar
import fcntl
import asyncio
from collections import defaultdict
import multiprocessing as mp

QUERY_DTYPE = np.int32

# Inverts the given results on disk.
class Inverter:
    
    ROWS_PER_SHARD = 10000000

    def __init__(self, qdir, workdir):
        self.qdir = qdir
        self.workdir = workdir
        self.map_output_dir = os.path.join(workdir, 'map_output')
        self.reduce_output_dir = os.path.join(workdir, 'reduce_output')
        self.results_file = os.path.join(workdir, 'results.txt')
        self.max_id_file = os.path.join(workdir, 'max_id.txt')
        self.max_id = 0

    def get_max_id(self):
        max_id = 0
        for qfile in os.listdir(self.qdir):
            qf = os.path.join(self.qdir, qfile)
            s = os.path.getsize(qf)
            # mid is a size-1 array with the last (i.e., largest) id in the
            # query file.
            mid = np.fromfile(qf, dtype=QUERY_DTYPE,
                    offset = s - np.dtype(QUERY_DTYPE).itemsize)
            assert len(mid) == 1
            max_id = max(max_id, int(mid[0]))
        return max_id

    def reset(self):
        if os.path.isdir(self.workdir):
            shutil.rmtree(self.workdir)
        os.mkdir(self.workdir)
        os.mkdir(self.map_output_dir)
        os.mkdir(self.reduce_output_dir)
        self.max_id = self.get_max_id()
        open(self.max_id_file, 'w').write(str(self.max_id))
        max_shards = int(1 + (self.max_id / Inverter.ROWS_PER_SHARD))
        # Open file descriptors fo
        for i in range(max_shards):
            # Create all the sharded files ahead of time because we need to lock
            # them later.
            rf, qf = self.get_map_output_filename(i)
            open(qf, 'ab').close()
            open(rf, 'ab').close()
       
        self.num_map_outputs = max_shards

    def get_map_output_filename(self, shard):
        qf = os.path.join(self.map_output_dir, '%d.q' % shard)
        rf = os.path.join(self.map_output_dir, '%d.r' % shard)
        return rf, qf

    def get_reduce_output_filename(self, ix):
        return os.path.join(self.reduce_output_dir, '%d.out' % ix)

    def write_map_output(self, shard, rows, query_ids):
        # We only need to lock the row file
        rf, qf = self.get_map_output_filename(shard)
        rf_obj = open(rf, 'ab')
        fcntl.lockf(rf_obj, fcntl.LOCK_EX)
        rf_obj.write(rows.tobytes())
        open(qf, 'ab').write(query_ids.tobytes())
        fcntl.lockf(rf_obj, fcntl.LOCK_UN)
        rf_obj.close()

    def mapper(self, qfile):
        query_id = int(qfile.split('.')[0][1:])
        end_ix = 0
        shard_id = -1
        read_count = 0
        qfilename = os.path.join(self.qdir, qfile)
        # Incrementally reading ends up being much faster than reading the whole
        # thing into memory each time.
        ids = np.fromfile(qfilename,
                count=Inverter.ROWS_PER_SHARD, dtype=QUERY_DTYPE)
        read_count += len(ids)
        read_finished = len(ids) < Inverter.ROWS_PER_SHARD
        while len(ids) > 0:
            shard_id += 1
            if not read_finished and len(ids) < Inverter.ROWS_PER_SHARD:
                tmp = np.fromfile(qfilename, count=Inverter.ROWS_PER_SHARD,
                        dtype=QUERY_DTYPE,
                        offset=np.dtype(QUERY_DTYPE).itemsize * read_count)
                read_count += len(tmp)
                read_finished = len(tmp) < Inverter.ROWS_PER_SHARD
                ids = np.concatenate((ids, tmp))
            
            row_limit = (shard_id + 1) * Inverter.ROWS_PER_SHARD
            end_ix = np.searchsorted(ids, row_limit, side='left')
            if end_ix == 0:
                continue
            self.write_map_output(shard_id, ids[:end_ix],
                    np.array([query_id] * end_ix, dtype=QUERY_DTYPE))
            
            # Update the shard
            ids = ids[end_ix:]
            sys.stdout.flush()

    def write_reduce_result(self, filename, region_sizes):
        # Save the results in case we need to use them again
        with open(filename, 'w') as resfile:
            for qs, c in region_sizes.items():
                # Make sure there are no repeats
                assert (len(set(qs)) == len(qs))
                data = np.append(list(qs), c)
                resfile.write(','.join(data.astype('str')) + '\n')

    def read_reduce_result(self, resultfile):
        region_size = defaultdict(int)
        with open(resultfile) as resfile:
            for line in resfile:
                parts = line.strip().split(',')
                qs = tuple([int(q) for q in parts[:-1]])
                c  = int(parts[-1])
                region_size[qs] = c
        return region_size

    # The reducer index is the id of the files (one row id and one query) that
    # it will process.
    def reducer(self, map_output_ix):
        region_size = defaultdict(int)
        rf, qf = self.get_map_output_filename(map_output_ix)
        rows = np.fromfile(rf, dtype=QUERY_DTYPE)
        queries = np.fromfile(qf, dtype=QUERY_DTYPE)
        assert (len(rows) == len(queries))
        pts_to_query = defaultdict(list)
        for i in range(len(rows)):
            pts_to_query[rows[i]].append(queries[i])
        for r, qs in pts_to_query.items():
            region_size[tuple(sorted(qs))] += 1
        self.write_reduce_result(
                self.get_reduce_output_filename(map_output_ix),
                region_size)

    def aggregate(self):
        # Do the same aggregation over the reducer output
        region_size = defaultdict(int)
        bar = ShadyBar('Aggregating', max=self.num_map_outputs,
            suffix='%(index)d/%(max)d - %(elapsed)ds')
        for ix in range(self.num_map_outputs):
            region_size_tmp = self.read_reduce_result(
                    self.get_reduce_output_filename(ix))
            for qs, c in region_size_tmp.items():
                region_size[qs] += c
            bar.next()
        bar.finish()
        self.write_reduce_result(self.results_file, region_size)
        return region_size

    def run(self):
        if os.path.exists(self.results_file):
            # There's something in the cache, read it.
            print('Using cached result in working directory')
            regions = self.read_reduce_result(self.results_file)
            max_id = int(open(self.max_id_file).read())
            return regions, max_id
        print('No cached result found in working directory')
        self.reset()
        cpus = mp.cpu_count()
        pool = mp.Pool(processes=cpus)
        print('Starting mapreduce using %d cores' % cpus)
        qfiles = os.listdir(self.qdir)
        bar = ShadyBar('Mapping', max=len(qfiles),
            suffix='%(index)d/%(max)d - %(elapsed)ds')
        for _ in pool.imap_unordered(self.mapper, qfiles):
            bar.next()
        bar.finish()
        pool.terminate()
        pool = mp.Pool(processes=cpus)
        bar = ShadyBar('Reducing', max=self.num_map_outputs,
            suffix='%(index)d/%(max)d - %(elapsed)ds')
        for _ in pool.imap_unordered(self.reducer, range(self.num_map_outputs)):
            bar.next()
        bar.finish()

        pool.terminate()
        return self.aggregate(), self.max_id

def points_for_query(qfile):
    ids = np.fromfile(qfile, dtype=QUERY_DTYPE)
    return ids

def is_sorted(arr):
    sorted = True
    for i in range(1, len(arr)):
        if arr[i] < arr[i-1]:
            sorted = False
            break
    return sorted

def build_point_map_inmemory(qdir):
    # Map from points to the set of queries its in
    point_map = {}
    max_id = 0
    for qfile in os.listdir(qdir):
        query_id = int(qfile.split('.')[0][1:])
        ids = points_for_query(os.path.join(qdir, qfile))
        assert is_sorted(ids)
        max_id = max(max_id, max(ids))
        print('Query %d (%d)' % (query_id, len(ids)))
        for i in ids:
            if i not in point_map:
                point_map[i] = []
            point_map[i].append(query_id)
    print('Max id: %d', max_id)
    print('Finished parsing all queries')

    # Take the set of points, turn it into a deterministic tuple, and hash each one, maintaining a counter of points per hash key.
    all_intersections = {}
    for _, v in point_map.items():
        s = tuple(sorted(v))
        if s not in all_intersections:
            all_intersections[s] = 0
        all_intersections[s] += 1

    print('Found %d intersections' % len(all_intersections))
    print('Region size (# points) @ [0, 10, 50, 90, 100]:',
            np.percentile(list(all_intersections.values()), [0, 10, 50, 90, 100]))
    return all_intersections, max_id
