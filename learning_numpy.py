import numpy as np

print(np.__version__)

a = np.array([1,2,3])
print(a)
print(a.shape)
print(a.dtype)
print(a.ndim)
print(a.size)
print(a.itemsize)
print(a[0])
a[0] = 10
print(a)

b = a * np.array([2,0,2])
print(b)

l = [1,2,3]
a = np.array([1,2,3])

l = l * 2
a = a * 2

print(l)
print(a)
a = np.array([1,2,3])
a = np.sqrt(a)
print(a)
a = np.log(a)
print(a)

#dot product 
l1 = [1,2,3]
l2 = [4,5,6]
a1 = np.array([1,2,3])
a2 = np.array(l2)
dot = 0
for i in range(len(l1)):
    dot += l1[i]*l2[i]
print(dot)

dot = np.dot(a1,a2)
print(dot)

sum1 = a1*a2
dot = np.sum(sum1)
print(dot)

dot = a1 @ a2
print(dot)


#speed test
from timeit import default_timer as timer

a = np.random.randn(1000)
b = np.random.randn(1000)

A = list(a)
B = list(b)

T = 1000

def dot1():
    dot = 0
    for i in range(len(A)):
        dot = A[i]*B[i]
    return dot 

def dot2():
    return np.dot(a,b)

start = timer()
for t in range(T):
    dot1()
end = timer()
t1 = end-start

start = timer()
for t in range(T):
    dot2()
end = timer()
t2 = end-start

print("list calculation", t1)
print('np.dot', t2)
print('ratio', t1/t2)

#multidimensional arrays

a = np.array([[1,2], [3,4]])
print(a)
print(a.shape)

print(a[0][0])
print(a.T)
print(np.linalg.inv(a))
print(np.linalg.det(a))
#overloading 
c = np.diag(a)
print(np.diag(c))

#slicing
a = np.array([[1,2,3,4], [5,6,7,8]])
print(a)
b = a[0,:]
print(b)
b = a[0,1:3]
print(b)
b = a[:,0]
print(b)
b = a[-1,-1]
print(b)

a = np.array([[1,2], [3,4], [5,6]])
print(a)
bool_idx = a > 2
print(bool_idx)
print(a[bool_idx])
print(a[a > 2])

b = np.where(a>2, a, -1)
print(b)

a = np.array([10,19,30,41,50,61])
print(a)
b = [1,3,5]
print(a[b])

even = np.argwhere(a%2==0).flatten()
print(a[even])

a = np.arange(1,7)
print(a)
print(a.shape)

b = a.reshape((2,3))
print(b)
print(b.shape)
b = a[np.newaxis, :]
print(b.shape)
print(b)
b = a[:, np.newaxis]
print(b)


#append or concatenate
a = np.array([[1,2], [3,4]])
print(a)
b = np.array([[5,6]])
c = np.concatenate((a,b), axis=None)
print(c)
c = np.concatenate((a,b.T), axis=1)
print(c)

a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
#hstack, vstack
c = np.hstack((a,b))
print(c)
c = np.vstack((a,b))
print(c)

#broadcasting

x = np.array([[1,2,3], [4,5,6], [1,2,3], [4,5,6]])
a = np.array([1,0,1])
y = x + a
print(y)

a = np.array([[7,8,9,10,11,12,13], [17,18,19,20,21,22,23]])
print(a)
print(a.sum(axis=None))
print(a.sum(axis=0))
print(a.sum(axis=1))
print(a.mean(axis=None))
print(a.var(axis=None))
print(a.std(axis=None))
x = np.array([1.0,2.0])
print(x)
print(x.dtype)
x = np.array([1.0,2.0], dtype=np.int64)
print(x.dtype)

a = np.array([1,2,3])
b = a
b[0] = 42
print(b)
print(a)

a = np.zeros((2,3))
print(a)

a = np.ones((2,3))
print(a)
print(a.dtype)

a = np.full((2,3), 5.0)
print(a)

a = np.eye(3)
print(a)

a = np.arange(20)
print(a)

a = np.linspace(0,10,5)
print(a)

#random numbers

a = np.random.random((3,2)) # 0-1
print(a)

a = np.random.randn(3,2) # mean=0 and var=1
#normal/Gaussian
print(a)

a = np.random.randn(1000)
print(a.mean(), a.var())

a = np.random.randint(3,10, size = (3,3))
print(a)

a = np.random.choice(5, size=10)
print(a)

#eigen values
a = np.array([[1,2], [3,4]])
eigenvalues, eigenvectors = np.linalg.eig(a)

print(eigenvalues)
print(eigenvectors) #column vector

# e_vec * e_val = A * e_vec

b = eigenvectors[:,0] * eigenvalues[0]
print(b)

c = a @ eigenvectors[:,0] 
print(b)

print(np.allclose(b,c))

#solving linear systems

A = np.array([[1,1], [1.5, 4.0]])
B = np.array([2200, 5050])

X = np.linalg.inv(A).dot(B)
print(X)

X = np.linalg.solve(A,B)
print(X)

#loading csv

#np.loadtxt, np.genfromtxt

data = np.loadtxt('spambase.csv', delimiter=',', dtype=np.float32)
print(data.shape)

data = np.genfromtxt('spambase.csv', delimiter=',', dtype=np.float32)
print(data.shape)