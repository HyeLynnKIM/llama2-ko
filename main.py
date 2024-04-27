import os
import torch
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--epochs', type=int, default=2, help='학습 에폭 수')
    parser.add_argument('--save_path', type=str, default='./', help='저장 경로')
    parser.add_argument('--batch_size', type=int, default=2, help='배치 사이즈')

    args = parser.parse_args()
