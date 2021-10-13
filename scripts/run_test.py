import argparse
import yaml
import os
from pathlib import Path
import shutil

BOOLEAN_ARGUMENTS = ["const-rooms", "const-goal", "var-envs"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--yaml_path', default='', help='full path to yaml file with run_configuration')
    args = parser.parse_args()
    return args


def get_yaml_params(yaml_path):
    with open (yaml_path, 'r') as File:
        list = yaml.load(File, yaml.SafeLoader)

    #Create initial command line to run
    cmd_line = ["python", "/home/nlp/New_nlp_rl/pytorch-a2c-ppo-acktr-gail/main.py"]
    for argument, value in list['run_config'].items():
        if argument in BOOLEAN_ARGUMENTS and value:
            cmd_line += ["--{}".format(argument)]
        elif argument not in BOOLEAN_ARGUMENTS:
            cmd_line += ["--{}".format(argument), "{}".format(value)]

    # Create directory for results
    Path(list['results_path']).mkdir(parents=True, exist_ok=True)

    shutil.copyfile(yaml_path, os.path.join(list['results_path'], "test_config.yml"))

    run_types = list['tests_config']['test_type']
    for type, test_details in run_types.items():
        if ("run" not in test_details.keys() or not test_details["run"]):
            pass
        else:
            run_general_test(type, cmd_line, test_details, list['results_path'])


def run_general_test(type, cmd_line, test_details, results_dir):
    # Get test cmd line for each test
    if type == "Regular_algo":
        return run_without_RP(cmd_line, test_details, results_dir)
    elif type == "Train_RP":
        return run_and_train_RP(cmd_line, test_details, results_dir)
    elif type == "Use_RP":
        return run_and_use_RP(cmd_line, test_details, results_dir)

def run_and_use_RP(cmd_line, test_details, results_dir):
    #Create use_RP directory
    new_results_path = os.path.join(results_dir, "use_RP")
    try:
        Path(new_results_path).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print ("results_path already exists")
        exit(1)

    print ("Created new use_RP directory in:\t{}".format(new_results_path))
    new_cmd_line = cmd_line + ["--save-dir", new_results_path]
    adding = ["--dijk-reward"] if test_details['reward_type'] == "Dijk" else []
    new_cmd_line = new_cmd_line  + adding
    new_cmd_line = new_cmd_line + ["--reward-pred", "--model-load", test_details['RP_path']]
    print ("\n Commiting the following cmd_line:\n{}".format(" ".join(new_cmd_line)))
    os.system (" ".join(new_cmd_line))
    return 0

def run_and_train_RP(cmd_line, test_details, results_dir):
    #Create Train_RP directory
    new_results_path = os.path.join(results_dir, "train_RP")
    try:
        Path(new_results_path).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print ("results_path: {} already exists".format(new_results_path))
        exit(1)

    print ("Created new train_RP directory in:\t{}".format(new_results_path))
    new_cmd_line = cmd_line + ["--save-dir", new_results_path]
    adding = ["--dijk-reward"] if test_details['reward_type'] == "Dijk" else []
    new_cmd_line = new_cmd_line  + adding
    new_cmd_line = new_cmd_line + ["--reward-pred", "--train-rp"]
    adding = ["--train-rp-online"] if test_details['train_method'] == "online" else []
    new_cmd_line = new_cmd_line  + adding
    print ("\n Commiting the following cmd_line:\n{}".format(" ".join(new_cmd_line)))
    os.system (" ".join(new_cmd_line))
    return 0

def run_without_RP(cmd_line, test_details, results_dir):
    #Create No_RP directory
    new_results_path = os.path.join(results_dir, "reg_algo")
    try:
        Path(new_results_path).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print ("results_path already exists")
        exit(1)

    print ("Created new reg_algo directory in:\t{}".format(new_results_path))
    new_cmd_line = cmd_line + ["--save-dir", new_results_path]
    adding = ["--dijk-reward"] if test_details['reward_type'] == "Dijk" else []
    new_cmd_line = new_cmd_line  + adding
    print ("\n Commiting the following cmd_line:\n{}".format(" ".join(new_cmd_line)))
    os.system (" ".join(new_cmd_line))
    return 0

def main(args):
    get_yaml_params(args.yaml_path)

if __name__ == '__main__':
    args = get_args()
    main(args)





