"""
Hachage statique simple sur disque.
- N buckets fixes.
- Chaque bucket est stocké dans un fichier (pickle).
- Si bucket plein (max_records_per_bucket), on crée une page d'overflow (chaînage).
Usage:
    ht = StaticHashTable('data_dir', num_buckets=8, max_records_per_bucket=4)
    ht.insert('key1', {'name':'a'})
    ht.get('key1')
    ht.delete('key1')
"""
import os
import pickle
from typing import Any, List

class BucketPage:
    def __init__(self, records=None, next_page=None):
        self.records = records or []  # list of (key, value)
        self.next_page = next_page    # filename of next page or None

class StaticHashTable:
    def __init__(self, path, num_buckets=16, max_records_per_bucket=4):
        self.path = path
        self.num_buckets = num_buckets
        self.max_records = max_records_per_bucket
        os.makedirs(path, exist_ok=True)
        # initialize bucket files
        for i in range(num_buckets):
            fname = self._bucket_filename(i)
            if not os.path.exists(fname):
                self._save_page(fname, BucketPage())

    def _bucket_filename(self, idx, suffix=""):
        return os.path.join(self.path, f"bucket_{idx}{suffix}.pkl")

    def _save_page(self, fname, page: BucketPage):
        with open(fname, "wb") as f:
            pickle.dump(page, f)

    def _load_page(self, fname) -> BucketPage:
        with open(fname, "rb") as f:
            return pickle.load(f)

    def _hash(self, key: str) -> int:
        # simple stable hash (can be replaced by murmur, etc.)
        return hash(key) % self.num_buckets

    def insert(self, key: str, value: Any):
        idx = self._hash(key)
        fname = self._bucket_filename(idx)
        page = self._load_page(fname)
        # try to update if exists, else insert
        cur_fname = fname
        while True:
            # update if exists
            for i, (k, v) in enumerate(page.records):
                if k == key:
                    page.records[i] = (key, value)
                    self._save_page(cur_fname, page)
                    return
            # if there is space insert
            if len(page.records) < self.max_records:
                page.records.append((key, value))
                self._save_page(cur_fname, page)
                return
            # otherwise follow next page or create one
            if page.next_page is None:
                # create new overflow page
                overflow_fname = cur_fname.replace(".pkl", f"_ovf.pkl")
                # ensure unique name by appending counter
                counter = 0
                candidate = overflow_fname
                while os.path.exists(candidate):
                    counter += 1
                    candidate = cur_fname.replace(".pkl", f"_ovf_{counter}.pkl")
                page.next_page = candidate
                self._save_page(cur_fname, page)
                page = BucketPage(records=[(key, value)])
                self._save_page(page.next_page, page)
                return
            else:
                cur_fname = page.next_page
                page = self._load_page(cur_fname)

    def get(self, key: str):
        idx = self._hash(key)
        fname = self._bucket_filename(idx)
        page = self._load_page(fname)
        while page:
            for k, v in page.records:
                if k == key:
                    return v
            if page.next_page is None:
                return None
            page = self._load_page(page.next_page)

    def delete(self, key: str) -> bool:
        idx = self._hash(key)
        fname = self._bucket_filename(idx)
        cur_fname = fname
        page = self._load_page(cur_fname)
        while True:
            for i, (k, v) in enumerate(page.records):
                if k == key:
                    page.records.pop(i)
                    self._save_page(cur_fname, page)
                    return True
            if page.next_page is None:
                return False
            cur_fname = page.next_page
            page = self._load_page(cur_fname)

    def debug_bucket(self, idx):
        fname = self._bucket_filename(idx)
        if not os.path.exists(fname):
            return None
        pages = []
        p = self._load_page(fname)
        cur = fname
        while p:
            pages.append((cur, list(p.records)))
            if p.next_page is None:
                break
            cur = p.next_page
            p = self._load_page(cur)
        return pages

if __name__ == "__main__":
    # test rapide
    ht = StaticHashTable('static_data', num_buckets=4, max_records_per_bucket=2)
    for i in range(10):
        ht.insert(f'k{i}', {'val': i})
    print(ht.get('k7'))
    print(ht.debug_bucket( hash('k7') % 4 ))
