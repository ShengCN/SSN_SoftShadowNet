import numpy as np

rand_ary = np.random.uniform(40,60,100)
output=''
for f in rand_ary:
    output = output + '{},'.format(int(f))

with open('random_fov.txt', 'a') as f:
    f.write(output)
