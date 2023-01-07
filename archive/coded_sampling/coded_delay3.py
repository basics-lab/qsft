from qspright.ReedSolomon import ReedSolomon

if __name__ == '__main__':
    n = 511
    t = 4
    q = 2
    rs = ReedSolomon(n, t, q)
    GF = rs.prime_field
    s = rs.s
    p = rs.get_parity_length()
    H1 = rs.H
    codeword = GF.Zeros(n)
    codeword[5] += GF(1)
    codeword[10] += GF(1)
    dec1 = rs.decode(codeword)
    D = rs.get_delay_matrix()
    syndrome1 = codeword.view(rs.field) @ H1[:, -n:].T
    syndrome2 = codeword @ D.T
    syndrome3 = rs.field.Vector([syndrome2[s * i:s * (i + 1)] for i in range(2 * t)])
    dec2 = rs.syndrome_decode(syndrome2)
