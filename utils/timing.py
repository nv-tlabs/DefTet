'''
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial
# 4.0 International License. https://creativecommons.org/licenses/by-nc/4.0/
'''

import json
import numpy as np
import torch
import time

class Timer(object):
    def __init__(self,
                 name,
                 max_samples,
                 ):

        if max_samples < 1:
            raise ValueError('Max samples must be greater than 0')

        self.name = name
        self.max_samples = max_samples
        self.next_sample_ptr = 0

        self.samples = []
        self.enabled = True
        self.started = False

    def start(self):
        if self.enabled:
            if self.started:
                raise RuntimeError(
                    'Timer "{}" already started'.format(self.name))

            torch.cuda.synchronize()
            self.time = time.time()
            self.started = True

        return self

    def end(self):
        if not self.enabled:
            return

        if not self.started:
            return

        self.started = False

        torch.cuda.synchronize()
        cur_time = time.time()
        elapsed = cur_time - self.time
        print('%s: Elapsed time: %.5f'%(self.name, elapsed))

        if self.next_sample_ptr >= len(self.samples):
            self.samples.append(elapsed)

        else:
            self.samples[self.next_sample_ptr] = elapsed

        self.next_sample_ptr = (self.next_sample_ptr + 1) % self.max_samples

        self.time = None

    def get_average(self):
        if not self.samples:
            return None

        if len(self.samples) <= 3:
            return sum(self.samples) / float(len(self.samples))

        else:
            return (sum(self.samples) -
                    min(self.samples) -
                    max(self.samples)) / \
                float(len(self.samples) - 2)

    def get_median(self):
        if not self.samples:
            return None

        return np.median(self.samples)

    def get_num_recorded_samples(self):
        return len(self.samples)

    def get_snapshot(self):
        return {
            'average': self.get_average(),
            'median': self.get_median(),
            'N': self.get_num_recorded_samples(),
        }


class TimerSnapshot(object):

    def __init__(self, timers):
        self.data = {
            name: timer.get_snapshot()
            for name, timer in timers.items()
        }

    def __str__(self):
        return json.dumps(self.data, sort_keys=True, indent=2)


class Timing(object):

    def __init__(self):
        self.timers = {}

    def reset(self, name=None):
        if name is None:
            self.timers = {}

        else:
            if name not in self.timers:
                raise ValueError('Timer "{}" not found'.format(name))

            del self.timers[name]

    def start(self,
              name,

              max_samples=500):

        if name not in self.timers:
            self.timers[name] = Timer(name,

                                      max_samples=max_samples)
        self.timers[name].start()

    def end(self, name):
        if name not in self.timers:
            return

        self.timers[name].end()

    def snapshot(self):
        return TimerSnapshot(self.timers)

    def save_snapshot(self, path):
        with open(path, 'w') as file:
            json.dump(self.snapshot().data, file, sort_keys=True, indent=2)