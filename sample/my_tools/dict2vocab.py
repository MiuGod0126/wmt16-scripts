import os
import sys
skip_words=['<s>','<pad>','</s>','<unk>']

def read_file(file):
    with open(file,'r',encoding='utf-8') as f:
        lines=f.readlines()
    return lines

def write_file(res,file):
    with open(file,'w',encoding='utf-8') as f:
        f.write(''.join(res))
    print(f'write to {file} success.')

def dic2vocab(in_file,out_file):
    dic=read_file(in_file)
    words=[d.strip().split(' ')[0] for d in dic]
    words=skip_words+words
    words=[w+'\n' for w in words]
    write_file(words,out_file)

if __name__ == '__main__':
    lang=sys.argv[1]
    folder=sys.argv[2]
    in_path=os.path.join(folder,f"dict.{lang}.txt")
    out_path=os.path.join(folder,f"vocab.{lang}")
    dic2vocab(in_file=in_path,out_file=out_path)
