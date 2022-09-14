from src.qspright.input_signal_long import LongSignal
import pickle
class PrecomputedSignal(LongSignal):

    def __init__(self, **kwargs):
        if kwargs.get("signal") is None:
            self._init_standard_params(**kwargs)
            self._init_random(**kwargs)
        else:
            self._init_given(**kwargs)

    def _init_given(self, **kwargs):
        self.noise_sd = kwargs.get("noise_sd")
        filename = kwargs.get("signal")
        with open(filename, 'rb') as f:
            self.Ms, self.Ds, self.q, self._signal_t = pickle.load(f)
            self.n, self.b = self.Ms[0].shape
            self.num_subsample = len(self.Ms)
            self.num_random_delays = len(self.Ds[0])
        f.close()
        if kwargs.get("transform"):
            with open(kwargs.get("transform"), 'rb') as f:
                self._signal_w, self.locq = pickle.load(f)



    def save_signal(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.Ms, self.Ds, self.q, self._signal_t), f)
        f.close()

    def save_transform(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._signal_w, self.locq), f)

