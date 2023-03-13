# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trust region optimization.

A lot of this is adapted from other's code.
See Schulman's Modular RL, wojzaremba's TRPO, etc.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf
import numpy as np


def var_size(v):
  return int(np.prod([int(d) for d in v.shape]))


def gradients(loss, var_list):
  grads = tf.gradients(ys=loss, xs=var_list)
  return [g if g is not None else tf.zeros(v.shape)
          for g, v in zip(grads, var_list)]

def flatgrad(loss, var_list):
  grads = gradients(loss, var_list)
  return tf.concat([tf.reshape(grad, [-1])
                    for (v, grad) in zip(var_list, grads)
                    if grad is not None], 0)


def get_flat(var_list):
  return tf.concat([tf.reshape(v, [-1]) for v in var_list], 0)


def set_from_flat(var_list, flat_theta):
  assigns = []
  shapes = [v.shape for v in var_list]
  sizes = [var_size(v) for v in var_list]

  start = 0
  assigns = []
  for (shape, size, v) in zip(shapes, sizes, var_list):
    assigns.append(v.assign(
        tf.reshape(flat_theta[start:start + size], shape)))
    start += size
  assert start == sum(sizes)

  return tf.group(*assigns)


class TrustRegionOptimization(object):

  def __init__(self, max_divergence=0.1, cg_damping=0.1):
    self.max_divergence = max_divergence
    self.cg_damping = cg_damping

  def setup_placeholders(self):
    self.flat_tangent = tf.compat.v1.placeholder(tf.float32, [None], 'flat_tangent')
    self.flat_theta = tf.compat.v1.placeholder(tf.float32, [None], 'flat_theta')

  def setup(self, var_list, raw_loss, self_divergence,
            divergence=None):
    self.setup_placeholders()

    self.raw_loss = raw_loss
    self.divergence = divergence

    self.loss_flat_gradient = flatgrad(raw_loss, var_list)
    self.divergence_gradient = gradients(self_divergence, var_list)

    shapes = [var.shape for var in var_list]
    sizes = [var_size(var) for var in var_list]

    # start = 0
    # tangents = []
    # for shape, size in zip(shapes, sizes):
    #   param = tf.reshape(self.flat_tangent[start:start + size], shape)
    #   tangents.append(param)
    #   start += size
    # assert start == sum(sizes)

    self.grad_vector_product = sum(
        tf.reduce_sum(input_tensor=g * t) for (g, t) in zip(self.divergence_gradient, tangents))
    self.fisher_vector_product = flatgrad(self.grad_vector_product, var_list)

    self.flat_vars = get_flat(var_list)
    self.set_vars = set_from_flat(var_list, self.flat_theta)

  def optimize(self, sess, feed_dict, raw_loss, policy_var, avg_self_kl):

    def calc_fisher_vector_product(tangent):
        # feed_dict[self.flat_tangent] = tangent
        # 调用flatgrad(raw_loss, var_list)
        # *fvp = sess.run(self.fisher_vector_product,
        #                 feed_dict=feed_dict)

        shapes = [var.shape for var in policy_var]
        sizes = [var_size(var) for var in policy_var]

        start = 0
        tangents = []
        for shape, size in zip(shapes, sizes):
            param = tf.reshape(tangent[start:start + size], shape)
            tangents.append(param)
            start += size
        assert start == sum(sizes)

        self.divergence_gradient = gradients(avg_self_kl, policy_var)
        self.grad_vector_product = sum(
            tf.reduce_sum(input_tensor=g * t) for (g, t) in \
                zip(self.divergence_gradient, tangents))       
        fvp = flatgrad(self.grad_vector_product, policy_var)

        fvp += self.cg_damping * tangent
        return fvp

    def calc_loss(theta):
      #调用set_from_flat(var_list, self.flat_theta)
    #   *sess.run(self.set_vars, feed_dict={self.flat_theta: theta})
      self.set_vars = set_from_flat(policy_var, theta)
      if self.divergence is None:
        
        # return sess.run(self.raw_loss, feed_dict=feed_dict), True
        return raw_loss_from_objective ,True
      else:
        # *raw_loss, divergence = sess.run(
        # *    [self.raw_loss, self.divergence], feed_dict=feed_dict)
        # *return raw_loss, divergence < self.max_divergence
        return raw_loss_from_objective, avg_kl < self.max_divergence

    #* old_theta = sess.run(self.flat_vars)
    old_theta = get_flat(policy_var)
    # 调用flatgrad(raw_loss, var_list)
    #* loss_flat_grad = sess.run(self.loss_flat_gradient,
    #                           feed_dict=feed_dict)

    loss_flat_grad = flatgrad(raw_loss, policy_var)

    step_dir = conjugate_gradient(calc_fisher_vector_product, -loss_flat_grad)

    shs = 0.5 * step_dir.dot(calc_fisher_vector_product(step_dir))
    lm = np.sqrt(shs / self.max_divergence)
    fullstep = step_dir / lm
    neggdotstepdir = -loss_flat_grad.dot(step_dir)

    # find optimal theta
    theta = linesearch(calc_loss, old_theta, fullstep, neggdotstepdir / lm)
    if self.divergence is not None:
       #*final_divergence = sess.run(self.divergence, feed_dict=feed_dict)
      final_divergence = cal_avg_kl
    else:
      final_divergence = None

    # set vars accordingly
    if final_divergence is None or final_divergence < self.max_divergence:
        self.set_vars = set_from_flat(policy_var, theta)
        #* sess.run(self.set_vars, feed_dict={self.flat_theta: theta})
    else:
        self.set_vars = set_from_flat(policy_var, old_theta)
        #* sess.run(self.set_vars, feed_dict={self.flat_theta: old_theta})


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
  p = b.copy()
  r = b.copy()
  x = np.zeros_like(b)
  rdotr = r.dot(r)
  for i in xrange(cg_iters):
    z = f_Ax(p)
    v = rdotr / p.dot(z)
    x += v * p
    r -= v * z
    newrdotr = r.dot(r)
    mu = newrdotr / rdotr
    p = r + mu * p
    rdotr = newrdotr
    if rdotr < residual_tol:
      break
  return x


def linesearch(f, x, fullstep, expected_improve_rate):
  accept_ratio = 0.1
  max_backtracks = 10

  fval, _ = f(x)
  for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
    xnew = x + stepfrac * fullstep
    newfval, valid = f(xnew)
    if not valid:
      continue
    actual_improve = fval - newfval
    expected_improve = expected_improve_rate * stepfrac
    ratio = actual_improve / expected_improve
    if ratio > accept_ratio and actual_improve > 0:
      return xnew

  return x
