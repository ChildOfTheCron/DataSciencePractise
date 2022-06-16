import neural_net_basic as nn
import matplotlib.pyplot as plt
import numpy as np

windows = 0
other = 1

#gelip_1 = [1, 1]
#gelip_2 = [1, 2]
#gelip_3 = [1, 3]
#goptar_1 = [2, 1]
#goptar_2 = [2, 2]
#goptar_3 = [2, 3]
#tarmac_1 = [3, 1]
#tarmac_2 = [3, 2]
#tarmac_3 = [3, 3]

training_input_vectors = np.array(
    [
        [1, 1],
        [2, 1],
        [3, 2],
        [2, 2],
        [3, 1],
        [1, 2],
        [3, 3],
        [1, 3],
        [2, 3],
        [2, 4],
        [2, 5],
    ]
)

targets = np.array([windows, other, windows, other, windows, windows, windows, windows, other, other, other])
learning_rate = 0.1
neural_network = nn.NeuralNetwork(learning_rate)
training_error = neural_network.train(training_input_vectors, targets, 10000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")

input_vector = [1,4]
final_vect = neural_network.predict(input_vector)
os_is = "other"
if final_vect < 0.5:
    os_is = "windows"
else:
    os_is = "mac"

print(final_vect)
print(os_is)
