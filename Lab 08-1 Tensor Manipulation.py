# Tensor 사용하는 방법.

import numpy as np
import tensorflow as tf
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim)  # rank(차원)
print(t.shape)  # shape(모양. 몇개의 element가 있는지)
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])  # slicing. (시작):(마지막+1)
print(t[:2], t[3:])

'''
1
(7,)
0.0 1.0 6.0
[ 2.  3.  4.] [ 4.  5.]
[ 0.  1.] [ 3.  4.  5.  6.]
'''

# tensor - Shape, Rank, Axis
# rank : [ 의 개수
# shape : rank의 수만큼 요소가 있음. rank가 4이면 shape은 (_,_,_,_) 형태.
# shape : ((0),(1),(2),(3))
# shape : (3) [의 끝까지 들어가 가장 안쪽 []에 들어가 있는 요소의 수.
# shape : (2) 가장 안쪽의 [(3)]가 한단계 밖의 [] 안에 몇개 들어가 있는 지.
# shape : (1) [(2)]가 몇개 있는 지.
# shape : (0) [(1)]가 몇개 있는 지.

t2 = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
print('\n', tf.shape(t2).eval())

# axis : 가장 바깥쪽의 []부터 축을 0으로 함.
# 가장 마지작, 즉 가장 안쪽의 축을 -1이라고도 함.

# 행렬 연산을 하고 싶은 경우, +,*등의 사칙연산을 사용하면 안됨.
# shape 이 다르더라도 연산을 하게 해주는 것 : Broadcasting

###
# Reduce mean : 평균을 구할 때는 항상 floating type으로 할 것.
# 축에 따라 결과가 달라진다. axis = -1을 사용하는 경우가 가장 많음.

# argmax : 가장 큰 요소의 인덱스(위치)를 구해준다.

# reshape : (2,2,3) 짜리를 (-1,3)으로 reshape하여라 라는 뜻은,
# 가장 안쪽 괄호에는 3개의 요소를 넣고 나머지는 알아서 하되, rank는 2로 하라는 뜻.

# squeeze, expand를 많이 사용함.
# squeeze : 각각의 요소만 뽑아 냄.
# expand : 차원을 확장함.

# one hot : 인덱스가 2이면 00100이라고 표현해주는 것.
# rank를 1개 더 expand하게 되는데, onehot하고 reshape하면 됨.

# casting : 형변환.

# stack : 축에 의해 쌓는 방법이 바뀌기도 함.

# ones and zeros like : 주어진 tensor와 같은 모양의 1로 채워진 또는 2로 채워진 tensor을 만들 수 있음.

# zip : 여러개를 한번에!