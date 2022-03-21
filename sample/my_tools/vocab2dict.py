'''
transfer paddle vocab to fariseq dict
paddle :
    <s>
    <pad>
    </s>
    <unk>
    a
    b
    c
fairseq:
    a 10
    b 9
    c 8
'''
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

def vocab2dic(in_file,out_file):
    vocab=read_file(in_file)[4:]
    freq=[100000-i for i in range(len(vocab))]
    lines=[f"{v.strip()} {f}\n" for v,f in zip(vocab,freq)] # 空格划分
    write_file(lines,out_file)

if __name__ == '__main__':
    lang=sys.argv[1]
    folder=sys.argv[2]
    in_path=os.path.join(folder,f"vocab.{lang}")
    out_path=os.path.join(folder,f"dict.{lang}.txt")
    vocab2dic(in_file=in_path,out_file=out_path)