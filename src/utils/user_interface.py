#!/usr/bin/env python

"""
https://github.com/mrahtz/learning-from-human-preferences/blob/master/utils.py
A simple CLI-based interface for querying the user about segment preferences.
"""

import logging
import queue
import time
from itertools import combinations
from multiprocessing import Queue, Process
from random import shuffle
import os
import sys
# import easy_tf_log
import numpy as np
from scipy.ndimage import zoom
import pyglet
import pickle
from filelock import FileLock
import glob

import tkinter
import tkinter.simpledialog as SD
# from tkinter import *

start_file_name = "src/utils/Start.np"
with open(start_file_name, 'rb') as F:
    START_FRAME = pickle.load(F)

class Segment:
    """
    A short recording of agent's behaviour in the environment,
    consisting of a number of video frames and the rewards it received
    during those frames.
    frames: initialize with frames
    rewards: initialize with rewards
    traj_mask: mask of valid state in order to train on windows in different sizes
    """
    # TODO: add observation ?
    def __init__(self, frames=[], obs=[], rewards=[], actions=[], user_reward=0):
        assert len(frames) == len(rewards) and len(obs) == len(rewards), "frames, obs, rewards must be with same length"
        self.frames = frames
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.user_reward = user_reward
        self.hash = self.finalise() if len(obs) > 0 else None

    def append(self, frame, reward):
        self.frames.append(frame)
        self.rewards.append(reward)

    def finalise(self, seg_id=None):
        if seg_id is not None:
            self.hash = seg_id
        else:
            # This looks expensive, but don't worry -
            # it only takes about 0.5 ms.
            self.hash = hash(np.array(self.frames).tostring())

    def change_user_reward(self, new_user_reward):
        self.user_reward = new_user_reward

    def __len__(self):
        return len(self.frames)


class PrefInterface:
    def __init__(self, synthetic_prefs, max_segs, max_user_r, log_dir=None,):
        '''
        max_user_r: expected user reward range. will be normalized to (-1, 1) accordingly.
                    if less than 0 - return original user reward.
        '''
        self.vid_q = Queue()
        if not synthetic_prefs:
            self.renderer = VideoRenderer(vid_queue=self.vid_q,
                                          mode=VideoRenderer.restart_on_get_mode,
                                          zoom=4)
        else:
            self.renderer = None
        self.synthetic_prefs = synthetic_prefs
        self.seg_idx = 0
        self.segments = []
        self.tested_pairs = set()  # For O(1) lookup
        self.max_segs = max_segs
        # easy_tf_log.set_dir(log_dir)
        self.max_user_r = max_user_r

    def stop_renderer(self):
        if self.renderer:
            self.renderer.stop()

    def set_pipes(self, seg_pipe, pref_pipe):
        self.seg_pipe = seg_pipe
        self.pref_pipe = pref_pipe

    def run(self):
        while len(self.segments) < 1:
            #print("Preference interface waiting for segments")
            time.sleep(5.0)
            self.recv_segments()

        while True:
            seg = None
            while seg is None:
                try:
                    # seg_pair = self.sample_seg_pair()
                    seg = self.segments.pop(0)
                except IndexError:
                    # print("Preference interface ran out of untested segments;"
                    #       "waiting...")
                    # If we've tested all possible pairs of segments so far,
                    # we'll have to wait for more segments
                    time.sleep(5.0)
                    self.recv_segments()


            #print("Querying preference for segments %s",seg.hash)

            if self.synthetic_prefs:
                seg.obs = np.stack(seg.obs)
                reward = np.sum(seg.rewards)
                actions = np.stack(seg.actions)
                traj_mask = np.ones(seg.obs.shape)  # TODO: Fix this
                self.ret_user(seg.obs, reward, actions, traj_mask)
            
            else:
                rewards = self.ask_user(seg)
                if rewards is not None:
                    # divide to segments according to received rewards
                    segments = len(rewards)
                    seg.obs = np.stack(seg.obs)
                    seg.actions = np.stack(seg.actions)
                    sub_ob_len = seg.obs.shape[0] // segments
                    for i in range(segments):
                        # split obs to segments
                        seg0 = i*sub_ob_len
                        seg1 = min((i+1)* sub_ob_len, seg.obs.shape[0])
                        ob = seg.obs[seg0 : seg1]
                        action = seg.actions[seg0 : seg1]
                        
                        r = rewards[i]

                        if self.max_user_r > 0:
                            # normalize r to range (-1, 1) and multipy in segment len
                            r = (r - self.max_user_r / 2) * 2 / self.max_user_r
                            r *= (seg1 - seg0)

                        traj_mask_ones = np.ones_like(seg.obs[ :(seg1 - seg0)])
                        traj_mask_zeros = np.zeros_like(seg.obs[(seg1 - seg0): ])
                        traj_mask = np.concatenate((traj_mask_ones, traj_mask_zeros))

                        dummy_ob = np.zeros_like(seg.obs[(seg1 - seg0) : ])
                        ob = np.concatenate((ob, dummy_ob))
                        dummy_action = np.zeros_like(seg.actions[(seg1 - seg0) : ])
                        action = np.concatenate((action, dummy_action))

                        assert (ob.shape == traj_mask.shape)
                        self.ret_user(ob, r, action, traj_mask)
                else:
                    print("WARNING: did not get user reward. skip window.")

            self.recv_segments()

    def recv_segments(self):
        """
        Receive segments from `seg_pipe` into circular buffer `segments`.
        """
        max_wait_seconds = 0.5
        start_time = time.time()
        n_recvd = 0
        segment = None
        while time.time() - start_time < max_wait_seconds:
            try:
                segment = self.seg_pipe.get(block=True, timeout=max_wait_seconds)
            except queue.Empty:
                segment = None
                continue
        
            if segment is not None:
                self.segments.append(segment)
                return

    def ask_user(self, seg):
        vid = []
        seg_len = len(seg)
        for t in range(seg_len):
            # border = np.zeros((84, 10), dtype=np.uint8)
            # -1 => show only the most recent frame of the 4-frame stack
            # frame = np.hstack((s1.frames[t][:, :, -1],
            #                    border,
            #                    s2.frames[t][:, :, -1]))
            frame = seg.frames[t]
            vid.append(frame)
        n_pause_frames = 7
        for _ in range(n_pause_frames):
            vid.append(np.copy(vid[-1]))
        self.vid_q.put(vid)

        while True:
            ROOT = tkinter.Tk()
            ROOT.withdraw()
            answer = SD.askstring(title="Input",
                prompt="How the agent was? Return a number between [0, {}] (Hint: {:.2f})".format(\
                    self.max_user_r if self.max_user_r > 0 else 'inf', np.sum(seg.rewards)))
            if answer is not None and (not answer == ""):
                try:
                    ans_list = answer.split()
                    rewards = [int(ans) for ans in ans_list]
                    if len(rewards) > 0:
                        break
                except:
                    print("not a num")
            else:
                print("Enter Reward")

        self.vid_q.put([np.zeros(vid[0].shape, dtype=np.uint8)])

        return rewards

    def ret_user(self, obs, reward, actions, traj_mask):
        self.pref_pipe.put((obs, reward, actions, traj_mask))


class PrefInterfaceFiles(PrefInterface):
    def __init__(self, synthetic_prefs, root_dir, max_segs, max_user_r, splits=1, worker_idx=0):
        '''
        splits - how many working users
        idx - index of woreker [0,1,...,splits-1]
        '''
        super(PrefInterfaceFiles, self).__init__(synthetic_prefs, max_segs, max_user_r)
        self.load_dir = os.path.join(root_dir, 'unlabeled')
        self.save_dir = os.path.join(root_dir, 'labeled')
        self.total_saved_windows = 0
        self.splits = splits
        self.worker_idx = worker_idx

    def recv_segments(self):
        """
        load segment from files instead of pipe
        """
        max_wait_seconds = 0.5
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            win_files = glob.glob(os.path.join(self.load_dir, 'win_*.pk'))
            load_win_files = []
            for file in win_files:
                n = int(file.split('win_')[-1].split('.pk')[0])
                if n % self.splits == self.worker_idx:
                    load_win_files.append(file)
            for file in load_win_files:
                print("[user_interface] load file  ", file)
                lock = FileLock(file + ".lock")
                with lock:
                    with open(file, 'rb') as handle:
                        seg_dict = pickle.load(handle)
                        self.segments.append(Segment(seg_dict['frames'],
                                                     seg_dict['obs'],
                                                     seg_dict['rewards'],
                                                     seg_dict['actions']))
                os.remove(file)

    def ret_user(self, obs, reward, actions, traj_mask):

        seg_dict = {
            'obs': obs,
            'user_reward': reward,
            'actions': actions,
            'traj_mask': traj_mask
        }
        label_num = self.total_saved_windows * self.splits + self.worker_idx
        seg_file = os.path.join(self.save_dir, 'win_{}.pk'.format(label_num))
        print("[user_interface] dump file ", seg_file)
        lock = FileLock(seg_file + ".lock")
        with lock:
            with open(seg_file, 'wb') as handle:
                pickle.dump(seg_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.total_saved_windows += 1

# Based on SimpleImageViewer in OpenAI gym
class Im(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        target_size = 300
        rep = target_size // arr.shape[0]
        arr = arr.repeat(rep, axis=0).repeat(rep, axis=1)

        if self.window is None:
            height, width = arr.shape[0], arr.shape[1]
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True

        # assert arr.shape == (self.height, self.width), \
        #     "You passed in an image with the wrong number shape"
        # import pdb; pdb.set_trace()
        arr = arr.astype('uint8')
        image = pyglet.image.ImageData(self.width, self.height,
                                       'RGB', arr.tobytes(), pitch=-self.width * 3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


class VideoRenderer:
    play_through_mode = 0
    restart_on_get_mode = 1

    def __init__(self, vid_queue, mode, zoom=1, playback_speed=1):
        assert mode == VideoRenderer.restart_on_get_mode or mode == VideoRenderer.play_through_mode
        self.mode = mode
        self.vid_queue = vid_queue
        self.zoom_factor = zoom
        self.playback_speed = playback_speed
        self.proc = Process(target=self.render)
        self.proc.start()
        self.FPS = 120 #90 #60

    def stop(self):
        self.proc.terminate()

    def render(self):
        v = Im()
        frames = self.vid_queue.get(block=True)
        print("video queue")
        print(frames[0].shape)
        print(len(frames))

        # add start frame
        #arr_init = np.expand_dims(255 * np.ones_like(frames[0]),axis=0)
        arr_init = np.flip(START_FRAME, axis=0)
        frames = [arr_init, arr_init, arr_init, arr_init] + frames
        t = 0
        while True:
            # Add a grey dot on the last line showing position
            # width = frames[t].shape[1]
            # fraction_played = t / len(frames)
            # x = int(fraction_played * width)
            # frames[t][-1][x] = 128

            # zoomed_frame = zoom(frames[t], self.zoom_factor)
            # v.imshow(zoomed_frame)
            v.imshow(np.flip(frames[t], axis=0))

            if self.mode == VideoRenderer.play_through_mode:
                # Wait until having finished playing the current
                # set of frames. Then, stop, and get the most
                # recent set of frames.
                t += self.playback_speed
                if t >= len(frames):
                    frames = self.get_queue_most_recent()
                    t = 0
                else:
                    time.sleep(1/self.FPS)
            elif self.mode == VideoRenderer.restart_on_get_mode:
                # Always try and get a new set of frames to show.
                # If there is a new set of frames on the queue,
                # restart playback with those frames immediately.
                # Otherwise, just keep looping with the current frames.
                try:
                    frames = self.vid_queue.get(block=False)
                    frames = [arr_init, arr_init, arr_init, arr_init] + frames
                    t = 0
                except queue.Empty:
                    t = (t + self.playback_speed) % len(frames)
                    time.sleep(1/60)


    def get_queue_most_recent(self):
        # Make sure we at least get something
        item = self.vid_queue.get(block=True)
        while True:
            try:
                item = self.vid_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                break
        return item

def start_pref_interface(seg_pipe, pref_pipe, max_segs, max_user_r, synthetic_prefs=False, log_dir=None):
    def f():
        # The preference interface needs to get input from stdin. stdin is
        # automatically closed at the beginning of child processes in Python,
        # so this is a bit of a hack, but it seems to be fine.
        sys.stdin = os.fdopen(0)
        pi.set_pipes(seg_pipe=seg_pipe, pref_pipe=pref_pipe)
        pi.run()

    # Needs to be done in the main process because does GUI setup work
    # prefs_log_dir = osp.join(log_dir, 'pref_interface')
    pi = PrefInterface(synthetic_prefs=synthetic_prefs,
                       max_segs=max_segs,
                       max_user_r=max_user_r,
                       log_dir=None)

    proc = Process(target=f, daemon=True)
    proc.start()
    return pi, proc

def saved_data_render(frames):
    '''
    frames - list of images to show
    '''
    v = Im()
    # arr_init = np.flip(START_FRAME, axis=0)

    if len(frames.shape) == 5:
        num_envs = (frames[0].shape[0])
        # add start frame
        #arr_init = np.expand_dims(255 * np.ones_like(frames[0]),axis=0)
        for env in range(num_envs):
            # env_frames = [arr_init, arr_init, arr_init, arr_init] + [frame[env] for frame in frames]
            env_frames = [frame[env] for frame in frames]
            frames_len = len(env_frames)
            for t in range(frames_len):
                # v.imshow(np.flip(env_frames[t], axis=0))
                v.imshow(env_frames[t])

                time.sleep(1 / 30)
    else:
        # env_frames = [arr_init, arr_init, arr_init, arr_init] + list(frames)
        env_frames =  list(frames)

        frames_len = len(env_frames)
        for t in range(frames_len):
            # v.imshow(np.flip(env_frames[t], axis=0))
            v.imshow(env_frames[t])
            time.sleep(1 / 60)  # 60 FPS, maybe change to 30


if __name__ == '__main__':
    print("run user interface with files")
    pi = PrefInterfaceFiles(synthetic_prefs=False,
                                max_segs=0,
                                max_user_r=10,
                            splits=2, worker_idx=1)
    pi.run()
    print("done")
