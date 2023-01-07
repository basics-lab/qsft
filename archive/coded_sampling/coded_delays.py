import unireedsolomon as rs
import math
import numpy as np

def print_hex(s):
    l = [ord(c) for c in s]
    print(":".join("{:02x}".format(num) for num in l))


if __name__ == '__main__':
    n = 30
    k = 15
    t_max = math.floor((n-k+1)/2)
    print(f"Max t:{t_max}")
    coder = rs.RSCoder(n, k, c_exp=8)
    H = coder.get_parity_check()
    H = np.array(H)
    m = '\x00' * k
    c = coder.encode(m)
    print(f"Codeword, len={len(c)}")
    print_hex(c)
    err = 1
    r = '\x01' * err + '\x00' * (n-err)
    print(f"Received, len={len(r)}")
    print_hex(r)
    s = coder.get_syndrome(r)
    print(f"Syndrome, len={len(s)}")
    print_hex(s)
    y = coder.decode(r)
    print("Decoded")
    print_hex(y[0])
