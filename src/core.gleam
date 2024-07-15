//// Core of learning algorithm
//// the tensor implementation details in inside learner.gleam
//// later on might use a flat-array tensor

import gleam/float
import gleam/function
import gleam/int
import gleam/list
import learner.{
  type Tensor, ListTensor, ScalarTensor, ext1, gradient_of, rank, shape,
  tensor1_map, tensor2_map, tensor_add, tensor_divide, tensor_exp, tensor_log,
  tensor_max, tensor_minus, tensor_multiply, tensor_multiply_2_1, tensor_sqr,
  tensor_sum, tensor_sum_cols, to_tensor, trefs,
}

//------------------------------------
// A-core
//------------------------------------

pub fn tensor_dot_product(w, t) {
  tensor_multiply(w, t) |> tensor_sum
}

pub fn tensor_dot_product_2_1(w, t) {
  tensor_multiply_2_1(w, t) |> tensor_sum
}

//------------------------------------
// B-layer-fns
//------------------------------------

// pub type Theta =
//   List(Tensor)

pub fn line(xs) {
  fn(theta) {
    let assert ListTensor([a, b]) = theta
    tensor_multiply(a, xs) |> tensor_add(b)
  }
}

pub fn quad(x) {
  fn(theta) {
    let assert ListTensor([a, b, c]) = theta

    tensor_multiply(a, tensor_sqr(x))
    |> tensor_add(tensor_multiply(b, x))
    |> tensor_add(c)
  }
}

pub fn linear_1_1(t) {
  fn(theta) {
    let assert ListTensor([a, b]) = theta

    tensor_dot_product(a, t)
    |> tensor_add(b)
  }
}

pub fn linear(t) {
  fn(theta) {
    let assert ListTensor([a, b]) = theta

    tensor_dot_product_2_1(a, t)
    |> tensor_add(b)
  }
}

pub fn plane(t) {
  fn(theta) {
    let assert ListTensor([a, b]) = theta

    tensor_dot_product(a, t)
    |> tensor_add(b)
  }
}

/// max normalization
///
/// The subtraction of the maximum value (t - max(t)) in the softmax function is a
/// numerical stability technique called "max normalization" or "log-sum-exp trick".
/// It's used to prevent numerical overflow and underflow issues when dealing with
/// very large or very small numbers in exponential calculations.
///
/// Here's why it's beneficial:
///
/// 1. Prevents overflow: Without this step, large input values could lead to
/// extremely large exponentials, causing overflow.
///
/// 2. Improves numerical stability: It ensures that at least one value in the
/// exponentiated result will be 1 (e^0 = 1), while others will be in the range [0,
/// 1].
///
/// 3. Doesn't change the result: Subtracting a constant from all elements before
/// applying softmax doesn't change the final probabilities, as softmax is
/// shift-invariant.
///
/// This technique allows the softmax function to handle a wider range of input
/// values reliably.
pub fn softmax(t) {
  fn(_theta) {
    let z = tensor_max(t) |> tensor_minus(t, _) |> tensor_exp
    tensor_sum(z) |> tensor_divide(z, _)
  }
}

pub fn signal_avg(t) {
  fn(_theta) {
    let assert [num_segments, ..] = shape(t) |> list.drop(rank(t) - 2)
    tensor_sum_cols(t)
    |> tensor_divide(num_segments |> int.to_float |> to_tensor)
  }
}

//------------------------------------
// C-loss
//------------------------------------
pub fn l2_loss(target) {
  fn(xs, ys) {
    fn(theta) {
      let pred_ys = theta |> target(xs)

      ys |> tensor_minus(pred_ys) |> tensor_sqr |> tensor_sum
    }
  }
}

pub fn cross_entropy_loss(target) {
  fn(xs, ys) {
    fn(theta) {
      let pred_ys = theta |> target(xs)
      let assert Ok(num_classes) = shape(ys) |> list.last

      tensor_log(pred_ys)
      |> tensor_dot_product(ys, _)
      |> tensor_divide(num_classes |> int.to_float |> to_tensor)
      |> tensor_multiply(-1.0 |> to_tensor)
    }
  }
}

pub fn kl_loss(target) {
  fn(xs, ys) {
    fn(theta) {
      let pred_ys = theta |> target(xs)

      tensor_divide(pred_ys, ys)
      |> tensor_log
      |> tensor_multiply(pred_ys)
      |> tensor_sum
    }
  }
}

// TODO: implement the logging utils
// with_recording

//------------------------------------
// D-gradient-descent
//------------------------------------
pub fn revise(f, revs, theta) {
  case revs {
    0 -> theta
    _ -> revise(f, revs - 1, f(theta))
  }
}

pub type Hyperparameters {
  Hyperparameters(
    revs: Int,
    alpha: Float,
    mu: Float,
    beta: Float,
    batch_size: Int,
  )
}

pub fn hp_new(revs revs, alpha alpha) {
  Hyperparameters(revs, alpha, 0.0, 0.0, 0)
}

pub fn hp_new_mu(hp hp: Hyperparameters, mu mu: Float) {
  Hyperparameters(
    revs: hp.revs,
    alpha: hp.alpha,
    mu: mu,
    beta: hp.beta,
    batch_size: hp.batch_size,
  )
}

pub fn hp_new_beta(hp hp: Hyperparameters, beta beta: Float) {
  Hyperparameters(
    revs: hp.revs,
    alpha: hp.alpha,
    mu: hp.mu,
    beta: beta,
    batch_size: hp.batch_size,
  )
}

pub fn hp_new_batch_size(hp hp: Hyperparameters, batch_size batch_size: Int) {
  Hyperparameters(
    revs: hp.revs,
    alpha: hp.alpha,
    mu: hp.mu,
    beta: hp.beta,
    batch_size: batch_size,
  )
}

pub fn gradient_descent(hp: Hyperparameters, inflate, deflate, update) {
  fn(obj, theta) {
    let tensor_update = hp |> update

    let f = fn(big_theta: Tensor) {
      tensor2_map(
        big_theta,
        gradient_of(obj, tensor1_map(big_theta, deflate)),
        tensor_update,
      )
    }

    tensor1_map(theta, inflate)
    |> revise(f, hp.revs, _)
    |> tensor1_map(deflate)
  }
}

//------------------------------------
// E-gd-common
//------------------------------------
pub fn zeros(x) {
  let f = fn(_) { 0.0 |> to_tensor } |> ext1(0)
  f(x)
}

pub fn smooth(decay_rate, average, g) {
  decay_rate *. average +. { 1.0 -. decay_rate } *. g
}

const epsilon = 1.0e-8

fn lift_update1(
  update: fn(Float, Float) -> Float,
) -> fn(Tensor, Tensor) -> Tensor {
  fn(p, g) { update(p |> tensor_to_float, g |> tensor_to_float) |> to_tensor }
}

fn lift_update2(
  update: fn(List(Float), Float) -> List(Float),
) -> fn(Tensor, Tensor) -> Tensor {
  fn(pa, g) {
    let assert ListTensor([p0, p1]) = pa
    update([p0, p1] |> list.map(tensor_to_float), g |> tensor_to_float)
    |> list.map(to_tensor)
    |> ListTensor
  }
}

fn lift_update3(
  update: fn(List(Float), Float) -> List(Float),
) -> fn(Tensor, Tensor) -> Tensor {
  fn(pa, g) {
    let assert ListTensor([p0, p1, p2]) = pa
    update([p0, p1, p2] |> list.map(tensor_to_float), g |> tensor_to_float)
    |> list.map(to_tensor)
    |> ListTensor
  }
}

//------------------------------------
// F-naked
//------------------------------------

fn naked_i(x) {
  function.identity(x)
}

fn naked_d(x) {
  function.identity(x)
}

fn naked_u(hp: Hyperparameters) {
  fn(p, g) { p -. hp.alpha *. g }
  |> lift_update1
}

pub fn naked_gradient_descent(hp: Hyperparameters) {
  gradient_descent(hp, naked_i, naked_d, naked_u)
}

//------------------------------------
// G-velocity
//------------------------------------
fn velocity_i(p: Tensor) {
  [p, zeros(p)] |> ListTensor
}

fn velocity_d(pa: Tensor) {
  let assert ListTensor([p0, ..]) = pa
  p0
}

/// momentum gradient descent
/// That's because we multiply the velocity v by a constant mu.
/// The resulting expression is analogous to the formula of momentum in physics.
fn velocity_u(hp: Hyperparameters) {
  fn(pa, g) {
    // pa: parameter P0 accompanied by its velocity P1 from the last revision
    let assert [p0, p1] = pa

    // μ * p1 - α * g
    let v = hp.mu *. p1 -. hp.alpha *. g

    // p0 + μ * p1 - α * g
    [p0 +. v, v]
  }
  |> lift_update2
}

pub fn velocity_gradient_descent(hp: Hyperparameters) {
  gradient_descent(hp, velocity_i, velocity_d, velocity_u)
}

//------------------------------------
// H-rms
//------------------------------------
// RMS: Root Mean Square
//
// adaptive learning rate gradient descent, uses a moving average of the squared
// gradients to smooth out the learning rate adaptation.
//
// use smooth to historically accumulate a modifier that is based on g.
fn rms_i(p: Tensor) {
  [p, zeros(p)] |> ListTensor
}

fn rms_d(pa: Tensor) {
  let assert ListTensor([p0, ..]) = pa
  p0
}

pub fn tensor_to_float(t: Tensor) -> Float {
  let assert ScalarTensor(s) = t
  s.real
}

fn rms_u(hp: Hyperparameters) {
  fn(pa, g) {
    let assert [p0, p1] = pa

    let r = smooth(hp.beta, p1, g *. g)
    let assert Ok(r_sqrt) = float.square_root(r)

    let alpha_hat = hp.alpha /. { r_sqrt +. epsilon }
    // p0 - (α / (r_sqrt + ϵ)) * g
    [p0 -. alpha_hat *. g, r]
  }
  |> lift_update2
}

pub fn rms_gradient_descent(hp: Hyperparameters) {
  gradient_descent(hp, rms_i, rms_d, rms_u)
}

//------------------------------------
// I-adam
//------------------------------------
fn adam_i(p: Tensor) {
  let zeroed = zeros(p)
  [p, zeroed, zeroed] |> ListTensor
}

fn adam_d(pa: Tensor) {
  let assert ListTensor([p0, ..]) = pa
  p0
}

// ADAM: adaptive moment estimation.
fn adam_u(hp: Hyperparameters) {
  fn(pa, g) {
    let assert [p0, p1, p2] = pa

    let r = smooth(hp.beta, p2, g *. g)
    let assert Ok(r_sqrt) = float.square_root(r)
    let alpha_hat = hp.alpha /. { r_sqrt +. epsilon }

    let v = smooth(hp.mu, p1, g)
    // The accompaniment v is known as the gradient's 1st moment.
    // The accompaniment r is known as the gradient's 2nd moment.
    // p0 - (α / (r_sqrt + ϵ)) * v
    [p0 -. alpha_hat *. v, v, r]
  }
  |> lift_update3
}

pub fn adam_gradient_descent(hp: Hyperparameters) {
  gradient_descent(hp, adam_i, adam_d, adam_u)
}

//------------------------------------
// J-stochastic
//------------------------------------
pub fn samples(n: Int, size: Int) -> List(Int) {
  list.range(0, n - 1) |> list.shuffle |> list.take(size)
}

pub fn sampling_obj(hp: Hyperparameters) {
  fn(expectant, xs, ys) {
    fn(theta) {
      let assert [n, ..] = shape(xs)
      let sample_indices = samples(n, hp.batch_size)

      let sampled_xs = trefs(xs, sample_indices)
      let sampled_ys = trefs(ys, sample_indices)

      theta |> expectant(sampled_xs, sampled_ys)
    }
  }
}
