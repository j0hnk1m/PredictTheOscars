from keras import backend as K
import tensorflow as tf
from itertools import product


def focal_loss(y_true, y_pred):
	gamma = 2.0
	alpha = 0.25
	pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
	return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def weighted_categorical_crossentropy(y_true, y_pred, weights):
	nb_cl = len(weights)
	final_mask = K.zeros_like(y_pred[:, 0])
	y_pred_max = K.max(y_pred, axis=1)
	y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
	y_pred_max_mat = K.equal(y_pred, y_pred_max)
	for c_p, c_t in product(range(nb_cl), range(nb_cl)):
		final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
	return K.categorical_crossentropy(y_pred, y_true) * final_mask

