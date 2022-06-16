import neural_net_basic as nn
import matplotlib.pyplot as plt
import numpy as np

training_input_vectors = np.array(
    [
        [3, 1.5],
        [2, 1],
        [4, 1.5],
        [3, 4],
        [3.5, 0.5],
        [2, 0.5],
        [5.5, 1],
        [1, 1],
    ]
)

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
learning_rate = 0.1
neural_network = nn.NeuralNetwork(learning_rate)
training_error = neural_network.train(training_input_vectors, targets, 10000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")

input_vector = [1,1]
final_vect = neural_network.predict(input_vector)
if final_vect < 0.5:
    final_vect = 0
else:
    final_vect = 1

print(final_vect)
