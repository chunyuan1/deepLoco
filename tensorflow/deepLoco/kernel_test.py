import numpy as np
import tensorflow as tf


'''test kernel'''
x = np.array([[[1,2],[3,4],[2,7],[2,5],[8,6]],[[5,6],[7,8],[5,3],[4,7],[5,2]]])
y = np.array([[[3,7],[2,9],[3,5],[7,5],[6,9]],[[7,4],[8,1],[2,4],[7,1],[2,9]]])
# print(np.repeat(3, 4))
# print(np.repeat(x, 2))
# print(x.shape)
x1=np.repeat(np.expand_dims(x,axis=2), 5, axis=2)
# print(x1)
# print("-------------------")
x2=np.repeat(np.expand_dims(y,axis=1), 5, axis=1)
# print(x2)
z=np.sum(np.abs(x1-x2),axis=3)
# print("-------------------")
print(z)
print("=====================")

x = tf.constant([[[1,2],[3,4],[2,7],[2,5],[8,6]],[[5,6],[7,8],[5,3],[4,7],[5,2]]])
y = tf.constant([[[3,7],[2,9],[3,5],[7,5],[6,9]],[[7,4],[8,1],[2,4],[7,1],[2,9]]])
with tf.Session() as sess:
    # print(sess.run(tf.tile([3], [4])))
    # print(sess.run(tf.squeeze(tf.reshape(tf.tile(tf.reshape(x, (-1, 1)), (1, 2)), (1, -1)))))
    # print(sess.run(tf.reshape(tf.tile(tf.reshape(x, (-1, 1)), (1, 3)), (2, -1))))
    x1 = tf.tile(tf.expand_dims(x, 2), [1, 1, 5, 1])
    # print(sess.run(x1))
    # print("-------------------")
    # print(sess.run(tf.shape(x1)))
    x2 = tf.tile(tf.expand_dims(y, 1), [1, 5, 1, 1])
    # print(sess.run(x2))
    # print("-------------------")
    z2 = tf.reduce_sum(tf.abs(x1-x2), 3)

    print(sess.run(z2))
    # print(sess.run(tf.shape(z2)))


'''test batch_dot'''
a = tf.placeholder(tf.float32, shape=(None, 3))
b = tf.placeholder(tf.float32, shape=(None, 3, 3))

c = tf.reduce_sum( tf.multiply( a, b ), 1, keep_dims=True )

with tf.Session() as session:
    print( c.eval(
        feed_dict={ a: [[1,1,1],[1,1,1],[1,1,1]],
                    b: [[[2,3,4],[5,6,7],[1,2,3]],
                        [[4,6,3],[2,5,8],[1,5,2]],
                        [[8,5,3],[7,3,5],[6,3,1]]] }
    ) )


a = tf.constant([[1,0,0],[2,0,0],[3,0,0]])

b = tf.constant([[[2,3,4],[5,6,7],[1,2,3]],
                 [[4,6,3],[2,5,8],[1,5,2]],
                 [[8,5,3],[7,3,5],[6,3,1]]])

a2 = tf.constant([[1,0,0],[2,1,0],[3,1,0]])
with tf.Session() as sess:
    a = tf.tile(tf.expand_dims(a,2),[1,1,3])

    c = tf.multiply(a,b)
    print(sess.run(c))
    print('------------')
    c = tf.reduce_sum(c, 1)
    print(sess.run(c))
    print('------------')
    c = tf.reduce_sum(tf.multiply(c,a2),1)
    print(sess.run(c))
    print('------------')
    print(sess.run(tf.reduce_sum(c)))


'''error calculate'''
a = tf.constant([[1,2,3],[1,0,0]])
b = tf.constant([[[1,1],[2,2],[3,3]],[[1,1],[2,2],[3,3]]])
with tf.Session() as sess:
    a = tf.tile(tf.expand_dims(a,2),[1,1,2])
    print(sess.run(tf.reduce_sum(tf.multiply(a,b))))

''''''

