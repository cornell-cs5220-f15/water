import matplotlib.pyplot as plt

time = [1.85321, 0.988055, 0.689624, 0.558584, 0.718732, 0.431079, 0.38988, 0.36192, 0.339751, 0.329152, 0.452864, 0.903567, 0.623262, 0.883788, 0.416284, 0.374226, 0.820113, 0.806807, 0.377407, 0.339908]
x = range(1,21)
plt.plot(x,time, linewidth=3)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel("number of threads", fontsize=20)
plt.ylabel("Time for the whole process", fontsize=20)
plt.title("Strong scaling for problem size = 200",fontsize=20)
plt.show()
