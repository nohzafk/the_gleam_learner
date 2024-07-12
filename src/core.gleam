//// Core of learning algorithm
//// the tensor implementation details in inside learner.gleam
//// later on might use a flat-array tensor

import gleam/int
import gleam/io
import gleam/list
import learner.{
  ListTensor, rank, shape, tensor_add, tensor_divide, tensor_exp, tensor_max,
  tensor_minus, tensor_multiply, tensor_multiply_2_1, tensor_sqr, tensor_sum,
  tensor_sum_cols, tensor_to_list, to_tensor,
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
  Hyperparameters(revs: Int, alpha: Float)
}

pub fn gradient_descent(hp: Hyperparameters, inflate, deflat, update) {
  todo
}
