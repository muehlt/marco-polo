# Fetching data from web source, to avoid git lfs

import requests
from tqdm import tqdm
import os
from pathlib import Path
import bz2
from halo import Halo
import shutil

class Fetcher:
    # NOTE: clear directory if keep-archives changed after download, skipping check if already downloaded everything
    keep_archives = False
    datapath = "../data/"
    #urls = ['http://www.cp.jku.at/datasets/LFM-2b/recsys22/listening_events.tsv.bz2',
    #        'http://www.cp.jku.at/datasets/LFM-2b/recsys22/tracks.tsv.bz2',
    #        'http://www.cp.jku.at/datasets/LFM-2b/recsys22/users.tsv.bz2']
    urls = ['https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip']

    def __fetch(self):
        for url in self.urls:
            filename = os.path.basename(url)
            uncomp_filename = Path(filename).stem
            if not os.path.exists(self.datapath + filename) and (self.keep_archives or not os.path.exists(self.datapath + uncomp_filename)):
                response = requests.get(url, stream=True)
                filesize = int(response.headers.get('content-length', 0))
                print("Downloading " + filename)
                progress_bar = tqdm(total=filesize, unit='iB', unit_scale=True)
                with open(self.datapath + filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))
                progress_bar.close()

            else: 
                print(f"File {filename} or directory {uncomp_filename} found locally, skipped download")

    def __extract_bz2(self):
        for url in self.urls:
            filename = os.path.basename(url)
            uncomp_filename = Path(filename).stem
            if not os.path.exists(self.datapath + uncomp_filename):
                spinner = Halo(text="Extracting " + filename, spinner='dots')
                spinner.start()
                bz2_file = bz2.BZ2File(self.datapath + filename)
                open(self.datapath + uncomp_filename, 'wb').write(bz2_file.read())
                spinner.succeed("Extracted " + filename)

                if not self.keep_archives:
                    spinner = Halo(text="Removing archive " + filename, spinner='dots')
                    spinner.start()
                    os.remove(self.datapath + filename)
                    spinner.succeed("Removed archive " + filename)
            else:
                print(f"File {uncomp_filename} found locally, skipped extraction")
    
    def __extract_zip(self):
        for url in self.urls:
            filename = os.path.basename(url)
            uncomp_filename = Path(filename).stem
            if not os.path.exists(self.datapath + uncomp_filename):
                spinner = Halo(text="Extracting " + filename, spinner='dots')
                spinner.start()
                shutil.unpack_archive(self.datapath + filename, self.datapath + uncomp_filename)
                spinner.succeed("Extracted " + filename)

                if not self.keep_archives:
                    spinner = Halo(text="Removing archive " + filename, spinner='dots')
                    spinner.start()
                    os.remove(self.datapath + filename)
                    spinner.succeed("Removed archive " + filename)
            else:
                print(f"Directory {uncomp_filename} found locally, skipped extraction")

    def fullSetup(self):
        self.__fetch()
        self.__extract_zip()
