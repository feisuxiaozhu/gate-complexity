import sympy as sym

def poly_builder(indices, trace):
    temp = 1
    for i in range(8):
        if indices[i] == '0':
            temp = temp* (1+Z_array[i])
        else:
            temp = temp*(1-Z_array[i])
    trace = sym.sympify(trace, rational=True)
    temp = temp*trace
    # temp = temp* (1/256) Don't forget this factor in the end!
    # print(sym.simplify(temp))
    return temp

Z_p0 = sym.Symbol('Z_p0')
Z_p1 = sym.Symbol('Z_p1')
Z_q0 = sym.Symbol('Z_q0')
Z_q1 = sym.Symbol('Z_q1')
Z_r0 = sym.Symbol('Z_r0')
Z_r1 = sym.Symbol('Z_r1')
Z_s = sym.Symbol('Z_s')
Z_t = sym.Symbol('Z_t')
Z_array = [Z_p0, Z_p1, Z_q0, Z_q1, Z_r0, Z_r1, Z_s, Z_t]

T5 = ['00010110', '00111110', '01000010', '01000110', '01001110', '01010010', '01110010', '11011110', '11110110','00011110', '00110110', '01010110', '01111110', '11000010', '11000110', '11001110', '11010010', '11110010']
TN1 = ['00000010', '00000110', '00001110', '00010010', '00110010', '01011110', '01110110', '11010110', '11111110']
TN5 = ['00000101', '00001101', '00010001', '00010101', '00110001', '00111101', '01000111', '01001111', '01010011', '01011111', '01110011', '01110111', '11000001', '11000011', '11010111', '11011101', '11110101', '11111111','00000111', '00001111', '00010011', '00011111', '00110011', '00110111', '01000001', '01000011', '01010111', '01011101', '01110101', '01111111', '11000101', '11001101', '11010001', '11010101', '11110001', '11111101']
T1 = ['00000001', '00000011', '00010111', '00011101', '00110101', '00111111', '01000101', '01001101', '01010001', '01010101', '01110001', '01111101', '11000111', '11001111', '11010011', '11011111', '11110011', '11110111']

result = 0

real = 0.5
for index in T5:
    result = result + sym.expand(poly_builder(index, real))

real = -0.5
for index in TN5:
    result = result + sym.expand(poly_builder(index, real))

real=1
for index in T1:
    result = result + sym.expand(poly_builder(index, real))

real = -1
for index in TN1:
    result = result + sym.expand(poly_builder(index, real))
    
print(result)

