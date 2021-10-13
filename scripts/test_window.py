
#==========================================================
# Script to run several offline trainings to test max
# window size
#==========================================================

import os
import sys
import subprocess
import time
import argparse
import pathlib

def main(args):
    for i in range(args.max_window,1,-1):
        save_dir = pathlib.Path(os.path.join(args.save_dir, "window_test_{}".format(i)))
        try:
            save_dir.mkdir(parents = True)
        except FileExistsError:
            print ("Results for window size {} already exists in the given directory".format(i))
            continue

        print ("Runnig train with window size {0} - results would be saved in {1}\n".format(i, save_dir)) 
        run_command = "python /home/nlp/New_nlp_rl/scripts/train_rp.py --env_rows {0} --env_cols {1} --save_dir {2} --data_path {3} --factor {4} --win_size {5}" .format(
          args.rows, args.cols, save_dir, args.data_path, args.pp_factor, i)

        run_command_list = run_command.split()

        log_file = os.path.join(save_dir, "train_log.txt")
        with open(log_file, "w+") as F:
            train_process = subprocess.Popen(run_command_list, stdout=F)
            return_code = train_process.wait()
            print ("Finished with window size {0} - results are in {1}\n".format(i, save_dir)) 


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_window', type=int, help='max window to test')
    parser.add_argument(
        '--data_path', help='full path to data directory')
    parser.add_argument(
        '--save_dir', help='full path to results directory')
    parser.add_argument(
        '--rows', type=int , default=30, help='rows size of data env')
    parser.add_argument(
        '--cols', type=int , default=30, help='cols size of data env')
    parser.add_argument(
        '--pp_factor', type=float , default=0.005, help='dijk factor used in the saving data script')
    return parser.parse_args()

if __name__ == '__main__':
    args = getArgs()
    main(args)
