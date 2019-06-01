
import os
import random
import math


f = open("list.txt")
test = open("testlist.txt", "w")
val = open("vallist.txt", "w")
count = 0
for line in f.readlines():
    if count % 2 == 0:
        test.write(line)
        test.flush()
    else:
        val.write(line)
        val.flush()
    count += 1
test.close()
val.close()
