from multiprocessing import Process
import numpy as np

import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def close_event():
    plt.close()  # timer calls this function after 3 seconds and closes the window

def show_rooms_traj(trajectory, ask_reward=False):
    print("show rooms trajectory")
    p = Process(target=_show_four_rooms_traj, args=[trajectory])
    p.start()
    if ask_reward:
        try:
            reward = float(input("Enter numeric reward: "))
        except:
            print("didn't get a number, return reward=0")
            reward = 0
    else:
        reward = 0
    time.sleep(1)
    return reward


def _show_four_rooms_traj(trajectory):
    '''
    prints a given trajectory, where:
    trajectory = [obs_0, obs_1, ..., obs_n]
    time between each state = 1 sec
    :param mode:
    :param close:
    :return:
    '''

    import time
    # plt.figure()
    traj_snapshots = []
    for obs in trajectory:
        if (obs != []):
            #obs = obs.numpy()

            if len(obs.shape) == 1:
                # flatten obs
                obs = 1.0 * (obs > 0.01) #TODO: check if helps
                l = int(obs.shape[0] / 3)
                sq_len = int(math.sqrt(l))
                player = obs[0:l].reshape(sq_len, sq_len).repeat(10, axis=0).repeat(10, axis=1)
                walls = obs[l:2*l].reshape(sq_len, sq_len).repeat(10, axis=0).repeat(10, axis=1)
                goal = obs[2*l:3*l].reshape(sq_len, sq_len).repeat(10, axis=0).repeat(10, axis=1)


            else:
                walls = obs[1, :, :]
                player = obs[0, :, :]
                goal = obs[2, :, :]

            color_walls_t = np.array([walls, walls, walls])
            color_walls = np.ones_like(color_walls_t) - color_walls_t
            color_player = np.array([np.zeros_like(player), player, player]) * -1
            color_goal = np.array([goal, goal, np.zeros_like(goal)]) * -1

            color_obs = color_player + color_walls + color_goal
            color_obs_im = color_obs.T
            traj_snapshots.append(color_obs_im)
            # plt.imshow(color_obs_im)
            # if first:
            #    plt.show(block=False)
            #    first = False
            # time.sleep(1)
            # plt.clf()
    # duration is in seconds

    # fig = plt.figure()
    # a = traj_snapshots[0]
    # im = plt.imshow(a)
    # anim = animation.FuncAnimation(fig, animate_func, fargs=(traj_snapshots, im), interval=200)  # in ms
    # # wait for time completion
    # plt.show()
    # return 0

    fig = plt.figure()
    # creating a timer object and setting an interval of 20 seconds
    timer = fig.canvas.new_timer(interval=60000)
    timer.add_callback(close_event)

    a = traj_snapshots[0]
    im = plt.imshow(a)
    anim = animation.FuncAnimation(fig, _animate_func, fargs=(traj_snapshots, im), interval=200)  # in ms
    # wait for time completion
    timer.start()
    plt.show()
    return 0



def _animate_func(i, snapshots, im):
    if (i >= len(snapshots)):
        im.set_array(snapshots[i %len(snapshots)])
    else:
        im.set_array(snapshots[i])
    return [im]
