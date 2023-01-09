from qsft.test_helper import TestHelper
from synt_src.synthetic_signal import SyntheticSubsampledSignal

class SyntheticHelper(TestHelper):
    def generate_signal(self, signal_args):
        return SyntheticSubsampledSignal(**signal_args)
