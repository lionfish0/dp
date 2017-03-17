import os
import random

from time import sleep
for resfile in range(2):
  for eps in [1.0,0.5,0.2,0.1,0.01]:
    for steps in [1,3,6,10]:
      print "Launching steps=%d eps=%0.3f" % (steps,eps)
      n = random.randint(0,1000000000000)
      os.system("python gen_paper_results.py %d %0.3f results%d.txt &" % (steps,eps,n))
      sleep(60.0)

#how to detect number of instances: ps a | grep 'python\ gen_paper_results.py' | wc -l

