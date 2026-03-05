import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt(r'C:\Users\student\Desktop\lv1_osu_pb\osnove_strojnog_ucenja_lv\lv2\data.csv', delimiter=',', skip_header=1)

gender = data[:, 0]
height = data[:, 1]
weight = data[:, 2]

# Broj osoba, shape izdvaja nultu dimenziju
print("Broj osoba:", data.shape[0])

plt.subplot(1,2,1)
plt.scatter(data[:,1], data[:,2])
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Odnos visine i mase")

plt.subplot(1,2,2)
plt.scatter(data[::50,1], data[::50,2])
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Svaka 50-ta osoba")

print("Minimalna visina:", np.min(data[:,1]))
print("Maksimalna visina:", np.max(data[:,1]))
print("Srednja visina:", np.mean(data[:,1]))

men_ind = (data[:,0] == 1)

print("Muskarci:")
print("Min:", np.min(data[men_ind,1]))
print("Max:", np.max(data[men_ind,1]))
print("Mean:", np.mean(data[men_ind,1]))

women_ind = (data[:,0] == 0)

print("Zene:")
print("Min:", np.min(data[women_ind,1]))
print("Max:", np.max(data[women_ind,1]))
print("Mean:", np.mean(data[women_ind,1]))

plt.show()
