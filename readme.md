# Readme



## /wikidata
this folder is gitignored but should contain the extraction wikipedia dump on your pc

### download
https://en.wikipedia.org/wiki/Wikipedia:Database_download
I used the torrent of the english version (24GB):
https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia

### preparation
Extraction takes in (60GB)

You'd need the deadsnakes apt source for the older version of python
    sudo add-apt-repository ppa:deadsnakes/ppa -y

For some reason you'd need this for the wikiextractor
    sudo apt install -y libbz2-dev

Apt update and install older version of python
     sudo apt install python3.10 python3.10-venv python3.10-dev

PyEnv to switch to the older version of python
    curl https://pyenv.run | bash

PyEnv will ask you to modify the following file
    source ~/.bashrc

PyEnv further setup for python 3.10, where tinyRag310 is a virtual environment name (venv)
    pyenv versions
    pyenv virtualenv 3.10.12 tinyRag310
    pyenv activate tinyRag310
    yenv local 3.10.12 # required for installing sentence-transformers 

Pip setup on python 3.10 (or use requirements.txt if present in prj)
    pip install --upgrade pip
    pip install wikiextractor

Nohup as in no hold up, the process will run in background, I had 6 processors:
    nohup python -m wikiextractor.WikiExtractor ~/Downloads/enwiki-20250301-pages-articles-multistream.xml.bz2 --json -o wiki_data --processes  6 --bytes 500M & tail -f nohup.out

### non pip installs / installs with issues
(tinyRag) python -m pip install sentence-transformers

## preprocess.py

This is the codefile that will train on the content of wiki_data

## main.py

This is the file that will answer questions via --query parameter
    python main.py --query "What are symptoms of a cold?"
