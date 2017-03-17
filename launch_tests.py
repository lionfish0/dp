import os
from time import sleep
for resfile in range(2):
  for eps in [1.0,0.5,0.2,0.1,0.01]:
    for steps in [1,3,6,10]:
      print "Launching steps=%d eps=%0.3f" % (steps,eps)
      os.system("python gen_paper_results.py %d %0.3f results.txt %d &" % (steps,eps,resfile,resfile))
      sleep(60.0)
