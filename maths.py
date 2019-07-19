from typing import List
import math
from typing import Tuple
from typing import Callable

from collections import Counter
import matplotlib.pyplot as plt

Vector = List[float]
Matrix = List[List[float]]

# Vector Functions
def add(v: Vector, w: Vector) -> Vector:
	#Adds corresponding elements
	assert len(v) == len(w), "vectors must be the same length"
	return [v_i + w_i for v_i, w_i in zip(v,w)]

def subtract(v: Vector, w: Vector) -> Vector:
	#Subtracts corresponding elements
	assert len(v) == len(w), "vectors must be the same length"
	return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors: List[Vector]) -> Vector:
	#Sums all corresponding elements
	assert vectors, "no vectors provided!"
	
	num_elements = len(vectors[0])
	assert all(len(v) == num_elements for v in vectors), "different sizes!"

	return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

def scalar_multiply(c: float, v: Vector) -> Vector:
	#Multiplies every element in c
	return [c * v_i for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
	#Computes the element-wise average
	n = len(vectors)
	return scalar_multiply(1/n, vector_sum(vectors))

def dot(v: Vector, w: Vector) -> float:
	#Computes v_1 * w_1 + .... + v_n + w_n
	assert len(v) == len(w), "vectors must be same length"
	return sum(v_i * w_i for v_i, w_i in zip(v,w))	

def sum_of_squares(v: Vector) -> float:
	#Returns v_1 * v_1 + ... v_n *  v_n
	return dot(v,v)

def magnitude(v: Vector) -> float:
	#Returns the magnitude (or length) of v
	return math.sqrt(sum_of_squares(v))

def squared_distance(v: Vector, w: Vector) -> float:
	#Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2
	return sum_of_squares(subtract(v,w))

def distance(v: Vector, w: Vector) -> float:
	return math.sqrt(squared_distance(v,w))

# Matrix Functions
def shape(A: Matrix) -> Tuple[int, int]:
	#Returns (# of rows of A, # of columns of A)
	num_rows = len(A)
	num_cols = len(A[0]) if A else 0 # Number of els in first row
	return num_rows, num_cols

def get_row(A: Matrix, i: int) -> Vector:
	#Returns the i-th row of A (as a Vector)
	return A[i]

def get_column(A: Matrix, j: int) -> Vector:
	#Returns the j-th column of A (as a Vector)
	return [A_i[j] for A_i in A]

def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
	#Returns a num_rows x num_cols matrix whose (i,j)-th entry is entry_fn(i,j)
	return [[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)]

def identity_matrix(n: int) -> Matrix:
	#Returns the n x n identity matrix
	return make_matrix(n, n, lambda i, j:1 if i == j else 0)

# Average Functions
def mean(xs: List[float]) -> float:
	return sum(xs) / len(xs)

def _median_odd(xs: List[float]) -> float:
	# If len(xs) is odd, the median is the middle element
	return sorted(xs)[len(xs) //2]
def _median_even(xs: List[float]) -> float:
	# If len(xs) is even, it's the average o the middle two elements
	sorted_xs = sorted(xs)
	hi_midpoint = len(xs) // 2 # eg. langth = 4 high midpoint = 2
	return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint])/2
def median(v: List[float]) -> float:
	#Finds the middle most value of v
	return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)

# Vector Tests
# [5,7,9]
print(add([1,2,3],[4,5,6]))
# [1,2,3]
print(subtract([5,7,9],[4,5,6]))
# [16,20]
print(vector_sum([[1,2],[3,4],[5,6],[7,8]]))
# [2,4,6]
print(scalar_multiply(2, [1,2,3]))
# [3,4]
print(vector_mean([[1,2],[3,4],[5,6]]))
# 32
print(dot([1,2,3],[4,5,6]))
# 14
print(sum_of_squares([1,2,3]))
# 5
print(magnitude([3,4]))
# Matrix Tests
#(2,3)
print(shape([[1,2,3],[4,5,6]]))
print(identity_matrix(5))

num_friends = [12, 45,23,43,54,2,5,34,54,34,54,34,54,23,74,72,34,32,22,13,55,38]
friend_counts = Counter(num_friends)
xs = range(101)
ys = [friend_counts[x] for x in xs]
#plt.bar(xs,ys)
#plt.axis([0,101,0,25])
#plt.title("Histogram of Friend Counts")
#plt.xlabel("# of friends")
#plt.ylabel("# of people")
#plt.show()

print(mean(num_friends))
print(median(num_friends))
