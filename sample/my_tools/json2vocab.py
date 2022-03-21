import os
import sys
import json

def load_save(in_path,out_path):
    with open(in_path,'r',encoding='utf-8') as f:
        data=json.load(f)
        keys=list(data.keys())
        keys=['<s>','<pad>','</s>','<unk>']+keys[3:]

    with open(out_path,'w',encoding='utf-8') as f:
        f.write('\n'.join(keys))
    print(f'write to {out_path} success.')

if __name__ == '__main__':
    lang=sys.argv[1]
    folder=sys.argv[2]
    in_path=os.path.join(folder,f"train.bpe.{lang}.json")
    out_path=os.path.join(folder,f"vocab.{lang}")
    load_save(in_path,out_path)