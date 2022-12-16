"""
    jupylet/clock.py
    
    Copyright (c) 2022, Nir Aides - nir.8bit@gmail.com

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import functools
import asyncio
import inspect
import logging
import random
import math
import time
import sys

import moderngl_window as mglw

from .utils import trimmed_traceback


logger = logging.getLogger(__name__)


class Timer(mglw.timers.clock.Timer):
    
    @property
    def time(self) -> float:

        if self._start_time is None:
            return 0

        return super(Timer, self).time


class FakeTime(object):
    
    def __init__(self):
        self._time = 0
        
    def time(self):
        return self._time        

    def sleep(self, dt):
        self._time += dt


def setup_fake_time():
    
    mglw.timers.clock.time = FakeTime()
    return mglw.timers.clock.time


def setup_real_time():
    mglw.timers.clock.time = time


class Scheduler(object):
    
    def __init__(self, timer):
        
        self._timer = timer
        self._sched = {}
        
    def schedule_once(self, foo, delay, **kwargs):
        logger.info('Enter Scheduler.schedule_once(foo=%r, delay=%r, **kwargs=%r).', foo, delay, kwargs) 

        self.unschedule(foo)
        self._sched[self._timer.time + delay, foo] = ('once', self._timer.time, None, kwargs)
        
    def schedule_interval(self, foo, interval, **kwargs):
        logger.info('Enter Scheduler.schedule_interval(foo=%r, interval=%r, **kwargs=%r).', foo, interval, kwargs) 
        
        self.unschedule(foo)
        self._sched[self._timer.time + interval, foo] = ('interval', self._timer.time, interval, kwargs)  
        
    def schedule_interval_soft(self, foo, interval, **kwargs):     
        logger.info('Enter Scheduler.schedule_interval_soft(foo=%r, interval=%r, **kwargs=%r).', foo, interval, kwargs) 

        self.unschedule(foo)
        self._sched[self._timer.time + interval, foo] = ('soft', self._timer.time, interval, kwargs)
        
    def unschedule(self, foo):
        # Python functions should not be compared for equality, not for identity:
        # https://stackoverflow.com/questions/18216597/how-should-functions-be-tested-for-equality-or-identity
        self._sched = {k: v for k, v in self._sched.items() if k[1] != foo}
        
    def call(self):
        
        tim0 = self._timer.time
        reap = {k: v for k, v in self._sched.items() if k[0] <= tim0}
        self._sched = {k: v for k, v in self._sched.items() if k[0] > tim0}
        
        for k, v in reap.items():
            
            t, foo = k
            _type, t0, i, kwargs  = v
            
            t1 = self._timer.time

            try:
                foo(t1, t1 - t0, **kwargs)
            except:
                logger.error(trimmed_traceback())
                _type = 'once'
            
            if _type == 'once':
                continue
                
            v = _type, t1, i, kwargs
            t = t + i * math.ceil((t1 - t) / i)
            
            if _type == 'interval':
                self._sched[t, foo] = v
                
            if _type == 'soft':
                self._sched[t + random.gauss(0, i / 32), foo] = v

        return max(0, self.time2next())
    
    def time2next(self):
        if self._sched:
            return min(k[0] for k in self._sched) - self._timer.time
        return 0.5
        

class ClockLeg(object):

    def __init__(self, timer=None, **kwargs):

        super(ClockLeg, self).__init__()

        self.scheduler = Scheduler(timer)
        self.schedules = {}
        
    # TODO: handle errors so application does not exit on user errors.
    def sonic_live_loop2(self, times=0, **kwargs):
        """Schedule an async function to run repeatedly in a loop.

        The function is scheduled to run based on its name, so updating
        its definition in a Jupyter notebook will cause the new definition
        to replace the previous one.

        The new definition will kick in once the current iteration through 
        the loop completes.

        Args:
            times (int): the number of times to run through the loop, or
                indefinately if 0.
        """
        return self.schedule_once(0, times, sync=True, **kwargs)
    
    def sonic_live_loop(self, times=0, **kwargs):
        """Schedule an async function to run repeatedly in a loop.

        The function is scheduled to run based on its name, so updating
        its definition in a Jupyter notebook will cause the new definition
        to replace the previous one.

        The new definition will kick in immediately.

        Args:
            times (int): the number of times to run through the loop, or
                indefinately if 0.
        """
        return self.schedule_once(0, times, sync=False, **kwargs)
    
    def run_me(self, delay=0, **kwargs):
        """Schedule a function to run after the specified delay.

        The function is scheduled to run based on its name, so updating
        its definition in a Jupyter notebook will cause the new definition
        to replace the previous one.

        Args:
            delay (float): the schedule delay in seconds.
        """
        return self.schedule_once(delay, 1, False, **kwargs)
    
    def run_me_every(self, interval, **kwargs):
        """Schedule a function to run after the specified delay.

        The function is scheduled to run based on its name, so updating
        its definition in a Jupyter notebook will cause the new definition
        to replace the previous one.

        Args:
            delay (float): the schedule delay in seconds.
        """
        return self.schedule_interval(interval, **kwargs)
    
    def schedule_once(self, delay=0, times=1, sync=False, **kwargs):

        def schedule0(foo):
            
            async def fuu(ct, dt):

                sc = self.schedules[foo.__name__]

                try:
                    while True:
                        
                        spec = sc['spec']
                        kwargs = sc['kwargs']
                        f00 = sc['foo']

                        if 'ct' in spec.args:
                            kwargs['ct'] = ct

                        if 'dt' in spec.args:
                            kwargs['dt'] = dt

                        if 'ncall' in spec.args:
                            kwargs['ncall'] = sc['ncall']

                        await f00(**kwargs)
                        
                        dt = self.scheduler._timer.time - ct
                        ct = ct + dt

                        sc['ncall'] += 1

                        if sc['times'] > 0 and sc['times'] == sc['ncall']:
                            break

                except asyncio.CancelledError:
                    pass
                except:
                    sc['errors'] = trimmed_traceback()
                    logger.error(sc['errors'])

            @functools.wraps(foo)
            def bar(ct, dt, **kwargs):
                
                sc = self.schedules[foo.__name__]

                if inspect.isgeneratorfunction(foo):
                    
                    goo = sc.get('gen')
                    if goo is None:
                        goo = foo(ct, dt, **kwargs)
                        sc['gen'] = goo
                        delay = next(goo)

                    else:
                        delay = goo.send((ct, dt))
                    
                    if delay is not None:
                        self.scheduler.schedule_once(bar, delay, **kwargs)
                        
                elif inspect.iscoroutinefunction(foo):

                    sc['spec'] = inspect.getfullargspec(foo)
                    sc['errors'] = None
                    sc['kwargs'] = kwargs
                    sc['times'] = times
                    sc['ncall'] = 0
                    sc['foo'] = foo

                    task = asyncio.get_event_loop().create_task(fuu(ct, dt))
                    sc['task'] = task
                    
                else:
                    foo(ct, dt, **kwargs)
                
            if sync and inspect.iscoroutinefunction(foo):
                sc = self.schedules.get(foo.__name__, {}) 
                if 'task' in sc and sc['errors'] is None:
                    if sc['times'] == 0 or sc['times'] > sc['ncall']:
                        sc['spec'] = inspect.getfullargspec(foo)
                        sc['kwargs'] = kwargs
                        sc['times'] = times                    
                        #sc['ncall'] = -1
                        sc['foo'] = foo

                        return foo

            self.unschedule(foo)
            self.schedules.setdefault(foo.__name__, {})['func'] = bar
            self.scheduler.schedule_once(bar, delay, **kwargs)

            return foo

        if inspect.isroutine(delay): # @app.run_me - without ()
            foo , delay = delay, 0
            return schedule0(foo)

        if inspect.isroutine(times): # @app.sonic_live_loop - without ()
            foo , times = times, 0
            return schedule0(foo)

        return schedule0

    def schedule_interval(self, interval, **kwargs):
        logger.info('Enter ClockLeg.schedule_interval(interval=%r, **kwargs=%r).', interval, kwargs) 

        def schedule0(foo):

            if inspect.iscoroutinefunction(foo):
                raise TypeError('Coroutine functions can only be scheduled with schedule_once() and its aliases.')
                
            if inspect.isgeneratorfunction(foo):
                raise TypeError('Generator functions can only be scheduled with schedule_once() and its aliases.')
                
            self.unschedule(foo)
            self.schedules.setdefault(foo.__name__, {})['func'] = foo
            self.scheduler.schedule_interval(foo, interval, **kwargs)

            return foo

        return schedule0

    def schedule_interval_soft(self, interval, **kwargs):
        """Schedule a function to run every `interval` seconds.
        
        This method is similar to `schedule_interval`, except that the
        clock will move the interval out of phase with other scheduled
        functions so as to distribute CPU more load evenly over time.
        """
        def schedule0(foo):

            if inspect.iscoroutinefunction(foo):
                raise TypeError('Coroutine functions can only be scheduled with schedule_once() and its aliases.')
                
            if inspect.isgeneratorfunction(foo):
                raise TypeError('Generator functions can only be scheduled with schedule_once() and its aliases.')
                
            self.unschedule(foo)
            self.schedules.setdefault(foo.__name__, {})['func'] = foo
            self.scheduler.schedule_interval_soft(foo, interval, **kwargs)

            return foo

        return schedule0

    def unschedule(self, foo=None, **kwargs):
        """Unschedule a function so it will not be called again."""
        
        if foo is None:
            fname = inspect.stack()[kwargs.get('levels_up', 1)][3] 
        elif type(foo) is str:
            fname = foo
        else:
            fname = foo.__name__
            
        d = self.schedules.pop(fname, {})
        
        if 'func' in d:
            self.scheduler.unschedule(d.get('func'))
        
        if 'task' in d:
            d['task'].cancel()
 
