# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time


def print_info(*message):
    print('\033[96m', *message, '\033[0m')

class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.time_total = 0.
        self.switch_count = 0
    
    def on(self):
        assert self.start_time is None, "Timer {} is already turned on!".format(self.name)
        self.start_time = time.time()
        
    def off(self):
        assert self.start_time is not None, "Timer {} not started yet!".format(self.name)
        self.time_total += time.time() - self.start_time
        self.start_time = None
        self.switch_count += 1

    def mean(self):
        if self.switch_count > 0:
            return self.time_total / self.switch_count
        else:
            return 0.
    
    def report(self):
        print_info('Time report [{}]: average time {:.4f}s ({:.4f}s in total for {} times)'.format(
            self.name, self.mean(), self.time_total, self.switch_count))

    def clear(self):
        self.start_time = None
        self.time_total = 0.
        self.switch_count = 0

class TimeReporter:
    def __init__(self):
        self.timers = {}

    def add_timer(self, name):
        assert name not in self.timers, "Timer {} already exists!".format(name)
        self.timers[name] = Timer(name = name)
    
    def start_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].on()
    
    def end_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].off()
    
    def report(self, name = None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
        else:
            print_info("------------Time Report------------")
            for timer_name in self.timers.keys():
                self.timers[timer_name].report()
            print_info("-----------------------------------")
    
    def clear_timer(self, name = None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].clear()
        else:
            for timer_name in self.timers.keys():
                self.timers[timer_name].clear()
    
    def pop_timer(self, name = None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
            del self.timers[name]
        else:
            self.report()
            self.timers = {}

    def get_time(self, name, mean = False):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        if mean:
            return self.timers[name].mean()
        else:
            return self.timers[name].time_total