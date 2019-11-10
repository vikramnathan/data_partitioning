import numpy as np
import shutil
import sys
import os
from array import array
from progress.bar import ShadyBar
import fcntl
import asyncio
from collections import defaultdict

QUERY_DTYPE = np.int32

# Inverts the given results on disk.
class Inverter:
    
    ROWS_PER_FILE = 10000000

    def __init__(self, qdir, workdir):
        self.qdir = qdir
        self.workdir = workdir
        self.interm_files_rowids = []
        self.interm_files_qids = []
        self.results_file = 'results.txt'
        self.interm_files_lock = asyncio.Lock()
        self.max_id = 0

    # Not threadsafe, must be called with self.interm_files_lock locked
    def add_interm_file(self):
        s = len(self.interm_files_rowids)
        newfile_r = os.path.join(self.workdir, 'interm_rows_%d' % s)
        newfile_q = os.path.join(self.workdir, 'interm_queries_%d' % s)
        self.interm_files_rowids.append(newfile_r)
        self.interm_files_qids.append(newfile_q)

   # async def mapper(qfile):
   #     query_id = int(qfile.split('.')[0][1:])
   #     # ids is a sorted list
   #     ids = np.fromfile(qfile, dtype=QUERY_DTYPE)
   #     start_ix = 0
   #     end_ix = 0
   #     shard_id = 0
   #     while end_ix < len(ids):
   #         row_limit += (shard_id + 1) * ROWS_PER_FILE
   #         end_ix = np.searchsorted(ids, row_limit, side='left')
   #         if end_ix == start_ix:
   #             continue
   #         # Make the list long enough to cover the new block id
   #         if len(self.interm_files) <= shard_id:
   #             async with lock.acquire():
   #                 while len(self.interm_files) <= shard_id:
   #                     self.add_interm_file()
   #         # Lock the relevant file
   #         fcntl.lock(self.interm_file[shard_id].fileno(), fcntl.LOCK_EX |
   #                 fcntl.LOCK_NB)
   #         self.interm_file[shard_id].write()

    def mapper(self, qfile):
        query_id = int(qfile.split('.')[0][1:])
        # ids is a sorted list
        print('Mapping query file %d' % query_id)
        # Corrects for the case where there are some repeat row IDs
        end_ix = 0
        shard_id = -1
        read_count = 0
        qfilename = os.path.join(self.qdir, qfile)
        # Incrementally reading ends up being much faster than reading the whole
        # thing into memory each time.
        ids = np.fromfile(qfilename,
                count=Inverter.ROWS_PER_FILE, dtype=QUERY_DTYPE)
        read_count += len(ids)
        while len(ids) > 0:
            shard_id += 1
            if len(ids) < Inverter.ROWS_PER_FILE:
                tmp = np.fromfile(qfilename, count=Inverter.ROWS_PER_FILE,
                        dtype=QUERY_DTYPE,
                        offset=np.dtype(QUERY_DTYPE).itemsize * read_count)
                read_count += len(tmp)
                ids = np.concatenate((ids, tmp))
                self.max_id = max(self.max_id, ids[-1])
            
            row_limit = (shard_id + 1) * Inverter.ROWS_PER_FILE
            end_ix = np.searchsorted(ids, row_limit, side='left')
            if end_ix == 0:
                continue
            # Make the list long enough to cover the new block id
            while len(self.interm_files_rowids) <= shard_id:
                self.add_interm_file()
            open(self.interm_files_rowids[shard_id], 'ab').write(
                    ids[:end_ix].tobytes())
            open(self.interm_files_qids[shard_id], 'ab').write(
                    np.array([query_id] * end_ix, dtype=QUERY_DTYPE).tobytes())
            
            # Update the shard
            ids = ids[end_ix:]
            sys.stdout.write('.')
            sys.stdout.flush()
        print('')

    def mapper2(self, qfile):
        query_id = int(qfile.split('.')[0][1:])
        # ids is a sorted list
        # TODO(vikram): Do we have to read this incrementally if too large?
        ids = np.fromfile(os.path.join(self.qdir, qfile), dtype=QUERY_DTYPE)
        print('Mapping query file %d, size %d' % (query_id, len(ids)))
        # Corrects for the case where there are some repeat row IDs
        ids = np.unique(ids)
        start_ix = 0
        end_ix = 0
        shard_id = -1
        while end_ix < len(ids):
            shard_id += 1
            row_limit = (shard_id + 1) * Inverter.ROWS_PER_FILE
            end_ix = np.searchsorted(ids, row_limit, side='left')
            if end_ix == start_ix:
                continue
            # Make the list long enough to cover the new block id
            while len(self.interm_files_rowids) <= shard_id:
                self.add_interm_file()
            # TODO(vikram): Lock the relevant file
            nrows = end_ix - start_ix
            open(self.interm_files_rowids[shard_id], 'ab').write(
                    ids[start_ix:end_ix].tobytes())
            open(self.interm_files_qids[shard_id], 'ab').write(
                    np.array([query_id] * nrows, dtype=QUERY_DTYPE).tobytes())
            
            # Update the shard
            start_ix = end_ix
            sys.stdout.write('.')
            sys.stdout.flush()
        print('')
        self.max_id = max(self.max_id, ids[-1])


    # The reducer index is the id of the files (one row id and one query) that
    # it will process.
    def reducer(self):
        region_size = defaultdict(int)
        bar = ShadyBar('Reducing', max=len(self.interm_files_rowids))
        for ix in range(len(self.interm_files_rowids)):
            rows = np.fromfile(self.interm_files_rowids[ix], dtype=QUERY_DTYPE)
            queries = np.fromfile(self.interm_files_qids[ix], dtype=QUERY_DTYPE)
            assert (len(rows) == len(queries))
            pts_to_query = defaultdict(list)
            for i in range(len(rows)):
                pts_to_query[rows[i]].append(queries[i])
            for r, qs in pts_to_query.items():
                region_size[tuple(sorted(qs))] += 1
            bar.next()
        bar.finish()
  
        # Save the results in case we need to use them again
        with open(self.results_file, 'a') as resfile:
            for qs, c in region_size.items():
                assert (len(set(qs)) == len(qs))
                data = np.append(qs, c)
                resfile.write(','.join(data.astype('str')) + '\n')

        return region_size

    def run(self):
        if os.path.isdir(self.workdir):
            shutil.rmtree(self.workdir)
        os.mkdir(self.workdir)
        qfiles = os.listdir(self.qdir)
        #bar = ShadyBar('Mapping', max=len(qfiles))
        for qfile in qfiles:
            self.mapper(qfile)
        #    bar.next()
        #bar.finish()
        return self.reducer(), self.max_id

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
