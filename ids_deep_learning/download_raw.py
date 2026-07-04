import os
import glob
import time

raw_dir = 'data/raw/cicids2017'
csv_files = sorted(glob.glob(os.path.join(raw_dir, '*.csv')))

print(f"[*] Found {len(csv_files)} CSV files in {raw_dir} to download.")

for path in csv_files:
    reported_size = os.path.getsize(path)
    print(f"\n[*] Downloading: {os.path.basename(path)}")
    print(f"[*] Reported size: {reported_size / 1024 / 1024:.2f} MB")
    
    start_time = time.time()
    size = 0
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8 * 1024 * 1024) # 8MB chunks
            if not chunk:
                break
            size += len(chunk)
            print(f"Downloaded: {size / 1024 / 1024:.2f} MB", end='\r')
            
    print(f"\n[+] Completed! Read {size / 1024 / 1024:.2f} MB in {time.time() - start_time:.2f} seconds.")
