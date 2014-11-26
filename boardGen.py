import random
import sys

f = open('input', 'w')
a = []
b = int(sys.argv[1]) * int(sys.argv[1])

for i in range(0, b):
    if(random.random() < 0.3):
        f.write('1')
    else:
        f.write('0')

f.close()
