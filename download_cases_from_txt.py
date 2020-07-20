import os
import shutil
import multiprocessing

from tqdm import tqdm
from urllib.request import Request, urlopen
from urllib.error import HTTPError

DATA_DIR = "/home/james/data/a4c3d/mva_train"
REMOTE_PATH = "https://files.magiquant.com/scantensus-database-png-flat"
MAX_FRAMES = 10000

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def download_case(case):
    case = case.strip()  # Get rid of linebreaks etc.
    if '.png' in case:  # Remove frame number and file extension if present
        case = case.rsplit('-',1)[0]
    subdir1 = case.split('-', 1)[0]
    subdir2, subdir3 = case[3:5], case[5:7]

    localdir = os.path.join(DATA_DIR, case)
    if not os.path.exists(localdir):
        os.makedirs(localdir)

        for i in range(MAX_FRAMES):
            remote = f"{REMOTE_PATH}/{subdir1}/{subdir2}/{subdir3}/{case}-{i:04}.png"
            req = Request(remote, headers=HEADERS)

            try:
                with urlopen(req) as response:
                    localfile = os.path.join(localdir, f"{i:04}.png")
                    with open(localfile, 'wb') as out_file:
                        shutil.copyfileobj(response, out_file)
            except HTTPError as e:  # 404 - skip to next study
                if i == 0:
                    print(f"ERROR ON FIRST FRAME for {remote} - {e}")
                break


if __name__ == "__main__":

    with open(os.path.join(DATA_DIR, "cases.txt")) as f:
        cases = f.readlines()

    N_WORKERS = 1

    with multiprocessing.Pool(N_WORKERS) as p:
        for _ in tqdm(p.imap(download_case, cases), total=len(cases)):
            pass
