import os
import random
import psutil

from time import sleep
for resfile in range(30):
  for eps in [1000000,0.2,0.5,1.0]:
    for steps in [1]:
      for lengthscale in [0.02,0.125,0.781]:
          for gaussian_noise in [3.0,9.0,27.0,81.0]:      
              print "Launching steps=%d eps=%0.3f" % (steps,eps)
              n = random.randint(0,1000000000000)
              os.system("python gen_paper_results.py %d %0.3f %0.5f %0.5f results%d.txt &" % (steps,eps,lengthscale,gaussian_noise,n))
              sleep(5.0)

              while psutil.cpu_percent()>80: #wait until we get below 80% CPU to start another thread
                sleep(10.0)

              #while (True): #this will block until there are less than 20 instances of python
              #  a = int(os.popen("ps -A | grep 'python' | wc -l").read())
              #  print a
              #  if (a<20):
              #    break
              #  sleep(1)
          
#how to detect number of instances: ps a | grep 'python\ gen_paper_results.py' | wc -l
