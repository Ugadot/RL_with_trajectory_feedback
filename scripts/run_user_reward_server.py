import argparse
import os
import sys

master_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, master_dir)

from src.utils.user_interface import PrefInterfaceFiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_interface_dir', default='/home/nlp/tmp/', type=str,
                        help="Path to dir with labeled, unlabeled dir")
    parser.add_argument('--splits', default=1, type=int,
                        help='number of labelers working together')
    parser.add_argument('--worker', default=0, type=int,
                        help='unique number of worker [0,1,...,splits-1')
    args = parser.parse_args()

    print("run user interface with files")
    pi = PrefInterfaceFiles(synthetic_prefs=False,
                            root_dir=args.user_interface_dir,
                            max_segs=0,
                            max_user_r=10,
                            splits=args.splits, worker_idx=args.worker)
    pi.run()