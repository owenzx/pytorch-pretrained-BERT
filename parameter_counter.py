import sys
import os
import torch
import argparse


def main(opt):
    ckpt_path = os.path.join(opt.model_dir, 'best.th')
    m = torch.load(ckpt_path)
    transformer_count = 0
    others_count = 0

    for k, v in m.items():
        if "transformer" in k:
            transformer_count += v.flatten().shape[0]
        else:
            others_count += v.flatten().shape[0]

    print("TRANSFORMER PARA COUNT: %d" % transformer_count)
    print("OTHER PARA COUNT: %d" % others_count)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-model_dir', type=str, help='directory of the trained model')
    opt = parser.parse_args()
    main(opt)
