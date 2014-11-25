import random

f = open('input', 'w')
a = []

for i in range(0, 16):
    if(random.random() < 0.3):
        f.write('1')
    else:
        f.write('0')

f.close()
