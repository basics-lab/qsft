from src.qspright.test_helper import TestHelper
from src.qspright.synthetic_signal import SyntheticSubsampledSignal


class SyntheticHelper(TestHelper):
    def generate_signal(self, signal_args):
        return SyntheticSubsampledSignal(**signal_args)
