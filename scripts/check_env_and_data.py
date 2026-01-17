"""
Check environment and dataset placement for the FL-CMAPPS project.

Usage:
  python scripts/check_env_and_data.py

This script reports Python version, existence of required files under data/raw/,
and prints row counts for each C-MAPSS file found.
"""
import sys
import os
import platform
import hashlib
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, 'data', 'raw')

EXPECTED = [
    'train_FD001.txt', 'test_FD001.txt', 'RUL_FD001.txt',
    'train_FD002.txt', 'test_FD002.txt', 'RUL_FD002.txt',
    'train_FD003.txt', 'test_FD003.txt', 'RUL_FD003.txt',
    'train_FD004.txt', 'test_FD004.txt', 'RUL_FD004.txt',
]

def print_pyinfo():
    print('Python:', sys.version.splitlines()[0])
    print('Platform:', platform.platform())

def file_row_count(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return None

def md5(path, block=65536):
    h = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(block), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def main():
    print('--- Environment check ---')
    print_pyinfo()
    
    # Check if data/raw/ exists
    if not os.path.exists(DATA_RAW):
        print(f'\n*** data/raw/ directory does not exist! ***')
        print(f'Please create it and place C-MAPSS files there.')
        print(f'Expected path: {DATA_RAW}')
        sys.exit(2)
    
    print('\n--- Dataset files check (data/raw/) ---')
    missing = []
    for fname in EXPECTED:
        path = os.path.join(DATA_RAW, fname)
        if os.path.exists(path):
            rows = file_row_count(path)
            checksum = md5(path)
            print(f'FOUND: {fname} | rows={rows} | md5={checksum[:8] if checksum else "-"}')
        else:
            print(f'MISSING: {fname}')
            missing.append(fname)

    # Also list any other files in data/raw/ for user visibility
    try:
        others = [f for f in os.listdir(DATA_RAW) if f not in EXPECTED]
        if others:
            print('\nOther files in data/raw/:')
            for f in others:
                print(' -', f)
    except Exception:
        pass

    print('\nSummary:')
    if missing:
        print(f'  Missing files: {len(missing)}. Please add them to data/raw/.')
        sys.exit(2)
    else:
        print('  All expected dataset files present.')
        sys.exit(0)

if __name__ == '__main__':
    main()
