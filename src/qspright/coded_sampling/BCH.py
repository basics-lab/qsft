import galois
from galois._codes._bch import decode_jit
import numpy as np
import math

class BCH(galois.BCH):

    def syndrome_decode(self, syndrome):
        # Invoke the JIT compiled function
        s = math.ceil(math.log2(self.n))
        codeword = galois.GF2.Zeros(self.n)
        syndrome_qs = self.field.Vector([syndrome[s * i:s * (i + 1)] for i in range(2 * self.t)])
        dec_codeword, n_errors = decode_jit(self.field)(codeword[:, np.newaxis].T, syndrome_qs[:, np.newaxis].T, self.t, int(self.field.primitive_element))
        return dec_codeword, n_errors