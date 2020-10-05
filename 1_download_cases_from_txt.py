import os
import shutil
import multiprocessing

from tqdm import tqdm
from itertools import repeat
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from lib.config import load_config

CONFIG = "./experiments/006.yaml"
MAX_FRAMES = 10000
HEADERS = {'User-Agent': 'Mozilla/5.0'}
FORCE_REDOWNLOAD = False

cfg = load_config(CONFIG)
remote_path = cfg['paths']['remote_path']


def download_case(case, data_path):
    case = case.strip()  # Get rid of linebreaks etc.
    if '.png' in case:  # Remove frame number and file extension if present
        case = case.rsplit('-',1)[0]
    subdir1 = case.split('-', 1)[0]
    subdir2, subdir3 = case[3:5], case[5:7]

    case_folder = os.path.join(data_path, case)
    if not os.path.exists(case_folder):
        exists = False
        os.makedirs(case_folder)
    else:
        exists = True

    if (not exists) or FORCE_REDOWNLOAD:
        print(f"Downloading {case_folder} as {exists} and {FORCE_REDOWNLOAD}")

        for i in range(MAX_FRAMES):
            remote = f"{remote_path}/{subdir1}/{subdir2}/{subdir3}/{case}-{i:04}.png"
            req = Request(remote, headers=HEADERS)

            try:
                with urlopen(req) as response:
                    localfile = os.path.join(case_folder, f"{i:04}.png")
                    with open(localfile, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
            except HTTPError as e:  # 404 - skip to next study
                if i == 0:
                    print(f"ERROR ON FIRST FRAME for {remote} - {e}")
                break
    else:
        print(f"Skipping {case_folder}")


if __name__ == "__main__":

    for cases_path, dest_path in zip((cfg['paths']['cases_train'], cfg['paths']['cases_test']),
                                     (cfg['paths']['data_train'], cfg['paths']['data_test'])):

        with open(cases_path) as f:
            cases = f.readlines()

        N_WORKERS = 1

        print(f"Found  {len(cases)} cases")

        with multiprocessing.Pool(N_WORKERS) as p:
            for _ in tqdm(p.starmap(download_case, zip(cases, repeat(dest_path))), total=len(cases)):
                pass
