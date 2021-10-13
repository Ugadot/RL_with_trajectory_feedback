import argparse
import matplotlib.pyplot as plt
import csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv_path', default='', help='full path to data saved in csv format')
    parser.add_argument(
        '--data_type', default='mean', help='should be either: mean, max, min, median. (default is mean)')
    parser.add_argument(
        '--model_learned', default='RP', help='should be either RP (for Reward Predictor) or PPO. default is RP')
    args = parser.parse_args()
    return args


def main(args):
    with open(args.csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    if args.model_learned == 'RP':
        # RP csv is as follows: ['max_model_n, min_model_n, mean_model_n, median_model_n']
        models_max_list = []
        models_min_list = []
        models_mean_list = []
        models_median_list = []
        episodes = list(range(len(data) - 1))
        episodes = [episode * 50 for episode in episodes]

        num_of_models = int((len(data[0])) / 4)
        print("Num of models are: {}".format(num_of_models))
        for i in range(num_of_models):
            start_index = i * 4
            models_max_list.append([float(result[start_index]) for result in data[1:]])
            models_min_list.append([float(result[start_index + 1]) for result in data[1:]])
            models_mean_list.append([float(result[start_index + 2]) for result in data[1:]])
            models_median_list.append([float(result[start_index + 3]) for result in data[1:]])

        if args.data_type == 'max':
            list_to_show = models_max_list
        if args.data_type == 'min':
            list_to_show = models_min_list
        if args.data_type == 'mean':
            list_to_show = models_mean_list
        if args.data_type == 'median':
            list_to_show = models_median_list

        for i in range(num_of_models):
            plt.plot(episodes, list_to_show[i])

        plt.xlabel('episode')
        plt.title(args.data_type + ' reward predictor Loss')
        plt.show()

    if args.model_learned == 'PPO':
        # PPO csv is as follows: ['num of episode', 'num_steps for all envs', FPS, num_of_episodes to avg from,
        # mean of PPO, median of PPO, min of PPO, max of PPO, dist_entropy, value_loss, action_loss]
        max_list = []
        min_list = []
        mean_list = []
        median_list = []
        episodes = [int(result[0]) for result in data[1:]]

        max_list = [float(result[7]) for result in data[1:]]
        min_list = [float(result[6]) for result in data[1:]]
        mean_list = [float(result[4]) for result in data[1:]]
        median_list = [float(result[5]) for result in data[1:]]

        if args.data_type == 'max':
            list_to_show = max_list
        if args.data_type == 'min':
            list_to_show = min_list
        if args.data_type == 'mean':
            list_to_show = mean_list
        if args.data_type == 'median':
            list_to_show = median_list

        plt.plot(episodes, list_to_show)
        plt.xlabel('episode')
        plt.title(args.data_type + ' reward')
        plt.show()

if __name__ == '__main__':
    args = get_args()
    main(args)





