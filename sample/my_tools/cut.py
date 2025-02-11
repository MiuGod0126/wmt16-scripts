import os
import sys
import time
import logging
from tqdm import tqdm
import jieba
logger=logging.getLogger('cut')
logger.setLevel(logging.DEBUG)

def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    return lines

def write_file(res,file):
    with open(file,'w',encoding='utf-8') as f:
        f.write(''.join(res))
    print(f'write to {file} success.')

res = []
def process(sent):
    sent=sent.strip()
    ls=jieba.lcut(sent)
    sent=' '.join(ls)+'\n'
    res.append(sent)
    return sent

if __name__ == '__main__':
    infile=sys.argv[1]
    outfile=sys.argv[2]
    lines=read_file(infile)
    for line in tqdm(lines):
        process(line)
    write_file(res,outfile)
