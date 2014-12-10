import random
import sys

f = open('input', 'w')
a = []
b = long(sys.argv[1]) * int(sys.argv[1])
c = long(sys.argv[1])

for j in range(0, c):
	for i in range(0, c):
	    if(random.random() < 0.3):
	        f.write('1')
	    else:
	        f.write('0')
	f.write('\n')
f.close()
