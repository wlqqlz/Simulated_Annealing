from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import datetime
import math
import pickle
import random
import sys
import time


def round_figures(x, n):
    """Returns x rounded to n significant figures."""
    return round(x, int(n - math.ceil(math.log10(abs(x)))))


def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)   # get hours and remainder
    m, s = divmod(s, 60)     # split remainder into minutes and seconds
    return '%4i:%02i:%02i' % (h, m, s)


class Annealer(object):

    """Performs simulated annealing by calling functions to calculate
    energy and make moves on a state.  The temperature schedule for
    annealing may be provided manually or estimated automatically.
    """

    __metaclass__ = abc.ABCMeta

    # defaults
    Tmax = 25000.0
    Tmin = 2.5
    steps_per_T = None
    #退火过程每个温度下迭代次数需人为给定：self.steps_per_T = 1000
    steps_to_coolT = 100
    #预设退火过程经过steps_to_coolT = 100个温度（包括Tmax和Tmin）
    copy_strategy = 'deepcopy'
    save_beststate_on_exit = False

    # placeholders
    best_state = None
    best_energy = None
    start = None

    def __init__(self, initial_state=None, load_state=None):
        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        elif load_state:
            self.load_state(load_state)
        else:
            raise ValueError('No valid values supplied for neither \
            initial_state nor load_state')


    def save_beststate(self, fname=None):
        """Saves state to pickle"""
        if not fname:
            date = datetime.datetime.now().strftime("%Y_%m_%d_%Hh_%Mm_%Ss")
            fname = date + "_energy_" + str(self.energy()) + ".state"
        with open(fname, "wb") as fh:
            pickle.dump(self.best_state, fh)


    def load_state(self, fname=None):
        """Loads state from pickle"""
        with open(fname, 'rb') as fh:
            self.state = pickle.load(fh)

    @abc.abstractmethod
    def move(self):
        """Create a state change"""
        # Necessary, process the state's change when call this function,
        # return the changed energy when the state's change happens.
        pass

    @abc.abstractmethod
    def energy(self):
        """Calculate state's energy"""
        # Necessary, required in the initial state's energy calculation at least.
        pass


    def set_schedule(self, schedule):
        """Takes the output from `auto` and sets the attributes
        """
        # set schedule, can manually or run auto function
        # steps: the total annealing steps
        self.Tmax = schedule['tmax']
        self.Tmin = schedule['tmin']
        self.steps_per_T = int(schedule['steps_per_T'])
        # 退火过程每个温度下迭代次数需人为给定：self.steps_per_T = 1000

    def copy_state(self, state):
        """Returns an exact copy of the provided state
        Implemented according to self.copy_strategy, one of

        * deepcopy: use copy.deepcopy (slow but reliable)
        * slice: use list slices (faster but only works if state is list-like)
        * method: use the state's copy() method
        """
        if self.copy_strategy == 'deepcopy':
            return copy.deepcopy(state)
        elif self.copy_strategy == 'slice':
            return state[:]
        elif self.copy_strategy == 'method':
            return state.copy()
        else:
            raise RuntimeError('No implementation found for ' +
                               'the self.copy_strategy "%s"' %
                               self.copy_strategy)

    def update(self, *args, **kwargs):
        """Wrapper for internal update.

        If you override the self.update method,
        you can chose to call the self.default_update method
        from your own Annealer.
        """
        self.default_update(*args, **kwargs)
        # update function means how to monitor the annealing process

    def default_update(self, step, T, E, acceptance, improvement):
        """Default update, outputs to stderr.

        Prints the current temperature, energy, acceptance rate,
        improvement rate, elapsed time, and remaining time.

        The acceptance rate indicates the percentage of moves since the last
        update that were accepted by the Metropolis algorithm.  It includes
        moves that decreased the energy, moves that left the energy
        unchanged, and moves that increased the energy yet were reached by
        thermal excitation.

        The improvement rate indicates the percentage of moves since the
        last update that strictly decreased the energy.  At high
        temperatures it will include both moves that improved the overall
        state and moves that simply undid previously accepted moves that
        increased the energy by thermal excititation.  At low temperatures
        it will tend toward zero as the moves that can decrease the energy
        are exhausted and moves that would increase the energy are no longer
        thermally accessible."""

        elapsed = time.time() - self.start
        if self.steps_per_T != None:
            remain = (self.steps_per_T * self.steps_to_coolT - step) * (elapsed / step)
            print('\r{Temp:12.5f}  {Energy:12.2f}   {Accept:7.2%}   {Improve:7.2%}  {Elapsed:s}  {Remaining:s}'
                  .format(Temp=T,
                          Energy=E,
                          Accept=acceptance,
                          Improve=improvement,
                          Elapsed=time_string(elapsed),
                          Remaining=time_string(remain)),
                  file=sys.stderr, end="")
            sys.stderr.flush()
        else:
            print('\r{Temp:12.5f}  {Energy:12.2f}   {Accept:7.2%}   {Improve:7.2%}  {Elapsed:s}  setting annealing schedule'
                  .format(Temp=T,
                          Energy=E,
                          Accept=acceptance,
                          Improve=improvement,
                          Elapsed=time_string(elapsed)),
                  file=sys.stderr, end="")
            sys.stderr.flush()

    def anneal(self):
        """Minimizes the energy of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        step_to_coolT = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        self.best_state = self.copy_state(self.state)
        self.best_energy = self.energy()
        prevState = self.copy_state(self.best_state)
        prevEnergy = self.best_energy
        self.E = self.best_energy

        print('\n Temperature        Energy    Accept   Improve     Elapsed   Remaining',
              file=sys.stderr)
        print('\r{Temp:12.5f}  {Energy:12.2f}                      {Elapsed:s}            '
              .format(Temp=self.Tmax,
                      Energy=self.best_energy,
                      Elapsed=time_string(time.time() - self.start)),
              file=sys.stderr, end="")
        sys.stderr.flush()

        # Attempt moves to new states
        while step_to_coolT < self.steps_to_coolT:
            trials = accepts = improves = 0
            T = self.Tmax * math.exp(Tfactor * step_to_coolT / (self.steps_to_coolT - 1))
            step_to_coolT += 1

            for _ in range(self.steps_per_T):
                step += 1
                dE = self.move()
                self.E += dE
                trials += 1
                if dE > 0.0 and math.exp(-dE / T) < random.random():
                    # Restore previous state
                    self.state = self.copy_state(prevState)
                    self.E = prevEnergy
                else:
                    # Accept new state and compare to best state
                    accepts += 1
                    prevState = self.copy_state(self.state)
                    prevEnergy = self.E
                    if dE < 0.0:
                        improves += 1
                    if self.E < self.best_energy:
                        self.best_state = self.copy_state(self.state)
                        self.best_energy = self.E

            self.state = self.copy_state(self.best_state)
            self.E = self.best_energy
            self.update(step, T, self.best_energy, accepts / trials, improves / trials)

        if self.save_beststate_on_exit:
            self.save_beststate()

        # Return best state and energy
        return self.best_state, self.best_energy

    def auto(self, minutes=None, steps_per_T=1000):
        # 退火过程每个温度下迭代次数auto预设为steps_per_T = 1000
        """Explores the annealing landscape and
        estimates optimal temperature settings.

        Returns a dictionary suitable for the `set_schedule` method.
        """

        def run(T, steps_per_T):
            """Anneals a system at constant temperature and returns the state,
            energy, rate of acceptance, and rate of improvement."""
            self.E = self.energy()
            prevState = self.copy_state(self.state)
            prevEnergy = self.E
            accepts, improves = 0, 0
            for _ in range(steps_per_T):
                dE = self.move()
                self.E = prevEnergy + dE
                if dE > 0.0 and math.exp(-dE / T) < random.random():
                    self.state = self.copy_state(prevState)
                    self.E = prevEnergy
                else:
                    accepts += 1
                    if dE < 0.0:
                        improves += 1
                    prevState = self.copy_state(self.state)
                    prevEnergy = self.E
                    if self.E < self.best_energy:
                        self.best_state = self.copy_state(self.state)
                        self.best_energy = self.E

            self.state = self.copy_state(self.best_state)
            self.E = self.best_energy
            return self.E, float(accepts) / steps_per_T, float(improves) / steps_per_T

        self.start = time.time()
        # Attempting automatic simulated anneal...
        # Find an initial guess for temperature

        self.E = self.energy()
        self.best_energy = self.E
        self.best_state = self.copy_state(self.state)
        T = 0.0
        step = 0

        print('\n Temperature        Energy    Accept   Improve     Elapsed   Remaining',
              file=sys.stderr)
        print('\r{Temp:12.5f}  {Energy:12.2f}                      {Elapsed:s}            '
              .format(Temp=T,
                      Energy=self.E,
                      Elapsed=time_string(time.time() - self.start)),
              file=sys.stderr, end="")
        sys.stderr.flush()

        while T == 0.0:
            step += 1
            dE = self.move()
            T = abs(dE)

        # Search for Tmax - a temperature that gives 98% acceptance
        self.E, acceptance, improvement = run(T, steps_per_T)

        step += steps_per_T
        while acceptance > 0.98:
            T = round_figures(T / 1.5, 2)
            self.E, acceptance, improvement = run(T, steps_per_T)
            step += steps_per_T
            self.update(step, T, self.E, acceptance, improvement)
        while acceptance < 0.98:
            T = round_figures(T * 1.5, 2)
            self.E, acceptance, improvement = run(T, steps_per_T)
            step += steps_per_T
            self.update(step, T, self.E, acceptance, improvement)
        Tmax = T

        # Search for Tmin - a temperature that gives 0% improvement
        while improvement > 0.0:
            T = round_figures(T / 1.5, 2)
            self.E, acceptance, improvement = run(T, steps_per_T)
            step += steps_per_T
            self.update(step, T, self.E, acceptance, improvement)
        Tmin = T

        # Calculate anneal duration
        elapsed = time.time() - self.start

        if minutes == None:
            duration_perT = steps_per_T
        else:
            duration_perT = round_figures(int(60.0 * minutes * step / elapsed), 2) / self.steps_to_coolT

        # Don't perform anneal, just return params
        return {'tmax': Tmax, 'tmin': Tmin, 'steps_per_T': duration_perT}
