//// learner
//// Tensor using list and recursive data structure implementation

import birl
import birl/duration
import parallel_map

import gleam/dict.{type Dict}
import gleam/dynamic.{type Dynamic}
import gleam/erlang.{type Reference}
import gleam/float
import gleam/function
import gleam/int
import gleam/io
import gleam/list
import gleam/pair
import gleam/result
import gleam/set
import gleam/string
import gleam_community/maths/elementary.{exponential, natural_logarithm}
import mat
import random_distribution.{random_normal}

//----------------------------
// Overview
//----------------------------
// - Tensors
// - Automaic Differentiation
// - Operator Extension
//      - pointwise extensions
//      - broadcast operations
// - Deep Learning Functions

//----------------------------
// Types
//----------------------------
pub const tolerace = 0.0001

pub fn tensor_id() {
  erlang.make_reference()
}

/// Tensors are implemented as nested lists and all scalars are duals.
/// scalar? - Tensors of rank 0
pub type Scalar {
  Scalar(id: Reference, real: Float, link: Link)
}

pub fn new_scalar(v: Float) -> Scalar {
  Scalar(tensor_id(), v, end_of_chain)
}

pub fn new_dual(v: Float, link: Link) {
  Scalar(tensor_id(), v, link)
}

pub fn from_scalar(s: Scalar) {
  Scalar(s.id, s.real, end_of_chain)
}

pub fn get_real(s: Scalar) {
  s.real
}

/// tensor? - Tensors of rank 0 or higher
pub type Tensor {
  ScalarTensor(Scalar)
  ListTensor(List(Tensor))
}

// shape? - The type (listof natural?) signifies the shape of a tensor. (virtual)
pub type Shape =
  List(Int)

// dual? - Duals

// gradient-state? - A hashtable from dual? to tensor?
pub type GradientState =
  Dict(Scalar, Float)

// link? - Links included in a dual. Defined as the type
// (-> dual? tensor? gradient-state? gradient-state?)
pub type Link =
  fn(Scalar, Float, GradientState) -> GradientState

pub fn end_of_chain(d: Scalar, z: Float, sigma: GradientState) {
  let g = dict.get(sigma, d) |> result.unwrap(0.0)
  dict.insert(sigma, d, z +. g)
}

// differentiable? - Either a dual?, or a (listof differentiable?).
// In the learner representation (vectorof differentiable?) is also considered to be differentiable?,
// but not in other representations.

// primitive-1? - A unary non-extended primitive. (virtual)

// primitive-2? - A binary non-extended primitive. (virtual)

// theta? - A list of tensors which forms a parameter set. (virtual)
pub type Theta =
  List(Tensor)

//----------------------------
// List functions
//----------------------------

pub fn refr(lst: List(a), n: Int) -> List(a) {
  lst |> list.drop(n)
}

//----------------------------
// Tensor functions
//----------------------------
pub fn tensor(lst: Dynamic) -> Tensor {
  case dynamic.classify(lst) {
    "Int" -> {
      use x <- result.map(lst |> dynamic.int)
      x |> int.to_float |> new_scalar |> ScalarTensor
    }
    "Float" -> {
      use x <- result.map(lst |> dynamic.float)
      x |> new_scalar |> ScalarTensor
    }
    "List" -> {
      use elements <- result.try(lst |> dynamic.list(dynamic.dynamic))
      let tensors = elements |> list.map(tensor)
      tensors |> ListTensor |> Ok
    }
    type_ -> {
      Error([
        dynamic.DecodeError(
          "a number or a list of numbers",
          type_ <> ": " <> string.inspect(lst),
          [],
        ),
      ])
    }
  }
  |> result.try_recover(fn(error) {
    io.debug(error)
    Error(error)
  })
  |> result.lazy_unwrap(fn() { panic as "Fail to decode as a Tensor" })
}

pub fn shape(t: Tensor) -> Shape {
  case t {
    ScalarTensor(_) -> []
    ListTensor([head, ..]) -> {
      [tlen(t), ..shape(head)]
    }
    _ -> panic as "Empty ListTensor"
  }
}

pub fn tlen(t: Tensor) -> Int {
  let assert ListTensor(lst) = t
  list.length(lst)
}

pub fn rank(t: Tensor) -> Int {
  t |> shape |> list.length
}

pub fn trefs(t: Tensor, b: List(Int)) -> Tensor {
  let assert ListTensor(lst) = t
  let index_set = set.from_list(b)

  lst
  |> list.index_map(fn(elem, index) { #(index, elem) })
  |> list.filter(fn(pair) { set.contains(index_set, pair.0) })
  |> list.map(fn(pair) { pair.1 })
  |> ListTensor
}

/// build tensor with shape, f is supplied with indexes
pub fn build_tensor(shape: Shape, f: fn(List(Int)) -> Float) -> Tensor {
  build_tensor_helper(f, shape, [])
}

fn build_tensor_helper(
  f: fn(List(Int)) -> Float,
  shape: Shape,
  idx: List(Int),
) -> Tensor {
  case shape {
    [s] ->
      list.range(0, s - 1)
      |> list.map(fn(i) {
        let new_idx = list.append(idx, [i])
        f(new_idx) |> to_tensor
      })
      |> ListTensor

    [s, ..rest] ->
      list.range(0, s - 1)
      |> list.map(fn(i) {
        let new_idx = list.append(idx, [i])
        build_tensor_helper(f, rest, new_idx)
      })
      |> ListTensor

    [] -> panic as "Shape cannot be empty"
  }
}

/// build tensor from other tensors, f is supplied with index and subtensor
pub fn build_tensor_from_tensors(
  others: List(Tensor),
  f: fn(List(#(Int, Tensor))) -> Tensor,
) -> Tensor {
  build_tensor_from_tensors_helper(f, others, [])
}

fn build_tensor_from_tensors_helper(
  f: fn(List(#(Int, Tensor))) -> Tensor,
  others: List(Tensor),
  items: List(#(Int, Tensor)),
) {
  case others {
    [t] -> {
      let assert ListTensor(lst) = t
      lst
      |> list.index_map(fn(a, i) {
        let new_items = list.append(items, [#(i, a)])
        f(new_items)
      })
      |> ListTensor
    }

    [t, ..rest] -> {
      let assert ListTensor(lst) = t
      lst
      |> list.index_map(fn(a, i) {
        let new_items = list.append(items, [#(i, a)])
        build_tensor_from_tensors_helper(f, rest, new_items)
      })
      |> ListTensor
    }

    [] -> panic as "others tensor cannot be empty"
  }
}

pub type TensorUnaryOp =
  fn(Tensor) -> Tensor

pub type TensorBinaryOp =
  fn(Tensor, Tensor) -> Tensor

//----------------------------
// Extended functions
//----------------------------
// base functions only work on tensors of a specific rank b
// base functions can be extended to work with tensors of any rank higher than b
//
// all the base and extended functions are differentiable

pub fn map_tensor(t: Tensor, f) -> Tensor {
  let assert ListTensor(lst) = t
  lst |> list.map(f) |> ListTensor
}

pub fn map_tensor2(t: Tensor, u: Tensor, f) -> Tensor {
  let assert ListTensor(lst_t) = t
  let assert ListTensor(lst_u) = u
  list.map2(lst_t, lst_u, f) |> ListTensor
}

/// apply f to a tensor t, f is a prim1 generated function
pub fn ext1(f, base_rank: Int) {
  fn(t: Tensor) {
    case rank(t) == base_rank {
      True -> f(t)
      _ -> map_tensor(t, ext1(f, base_rank))
    }
  }
}

/// apply f to 2 tensors t u, f is a prim2 generated function
pub fn ext2(f, n: Int, m: Int) {
  fn(t: Tensor, u: Tensor) {
    case rank(t) == n && rank(u) == m {
      True -> f(t, u)
      _ -> desc(ext2(f, n, m), n, t, m, u)
    }
  }
}

// descend operation g into tensors t of rank n, u of rank m
fn desc(g, n: Int, t: Tensor, m: Int, u: Tensor) {
  case rank(t), rank(u) {
    a, _ if a == n -> desc_u(g, t, u)
    _, b if b == m -> desc_t(g, t, u)
    a, b if a == b -> map_tensor2(t, u, g)
    a, b if a > b -> desc_t(g, t, u)
    a, b if a < b -> desc_u(g, t, u)
    _, _ ->
      panic as mat.format4(
        "Shapes are incompatible for ext2: {} and {} for min ranks {} and {}",
        shape(t),
        shape(u),
        n,
        m,
      )
  }
}

pub fn desc_t(g, t: Tensor, u: Tensor) {
  t |> map_tensor(fn(e) { g(e, u) })
}

pub fn desc_u(g, t: Tensor, u: Tensor) {
  u |> map_tensor(fn(e) { g(t, e) })
}

//
// Primitives in learner and nested-tensors
//
pub fn get_float(t: Tensor) {
  let assert ScalarTensor(s) = t
  s.real
}

pub fn get_scalar(t: Tensor) {
  let assert ScalarTensor(s) = t
  s
}

pub fn to_tensor(v: Float) {
  new_scalar(v) |> ScalarTensor
}

/// Constructs a differentiable function (known as a primitive) of one tensor
/// argument that invokes ρ-fn to compute the result of the application of the
/// primitive, and uses ∇-fn to find the gradient of the result with respect to the
/// argument provided to the primitive.
pub fn prim1(real_fn, gradient_fn) -> TensorUnaryOp {
  fn(t: Tensor) {
    new_dual(
      // real
      { real_fn |> lift_float1 |> lift_scalar1 }(t) |> get_float,
      // link
      fn(_d, z, sigma) {
        let s1 = get_scalar(t)
        let ga = gradient_fn(t |> get_float, z)

        sigma
        |> s1.link(s1, ga, _)
      },
    )
    |> ScalarTensor
  }
}

pub fn prim2(real_fn, gradient_fn) -> TensorBinaryOp {
  fn(t: Tensor, u: Tensor) {
    new_dual(
      // real
      { real_fn |> lift_float2 |> lift_scalar2 }(t, u) |> get_float,
      // link
      fn(_d, z, sigma) {
        let s1 = get_scalar(t)
        let s2 = get_scalar(u)
        let assert [ga, gb] = gradient_fn(t |> get_float, u |> get_float, z)

        sigma
        |> s1.link(s1, ga, _)
        |> s2.link(s2, gb, _)
      },
    )
    |> ScalarTensor
  }
}

fn lift_float1(f: fn(Float) -> Float) -> fn(Scalar) -> Scalar {
  fn(x) { x |> get_real |> f |> new_scalar }
}

fn lift_float2(f: fn(Float, Float) -> Float) -> fn(Scalar, Scalar) -> Scalar {
  fn(x, y) {
    f(x |> get_real, y |> get_real)
    |> new_scalar
  }
}

fn lift_scalar1(f: fn(Scalar) -> Scalar) -> TensorUnaryOp {
  fn(t: Tensor) {
    let assert ScalarTensor(a) = t
    f(a) |> ScalarTensor
  }
}

fn lift_scalar2(f: fn(Scalar, Scalar) -> Scalar) -> TensorBinaryOp {
  fn(ta: Tensor, tb: Tensor) {
    let assert ScalarTensor(a) = ta
    let assert ScalarTensor(b) = tb
    f(a, b) |> ScalarTensor
  }
}

//----------------------------
// Automaic Differentiation
//----------------------------
pub fn gradient_of(f: fn(Theta) -> Tensor, theta: Theta) -> Theta {
  let wrt = map_tensor_recursively(from_scalar, theta |> ListTensor)
  let assert ListTensor(wrt_lst) = wrt

  let assert ListTensor(lst) = gradient_once(f(wrt_lst), wrt_lst)
  lst
}

pub fn gradient_once(y, wrt) {
  let sigma = gradient_sigma(y, dict.new())
  map_tensor_recursively(
    fn(s) { sigma |> dict.get(s) |> result.unwrap(0.0) |> new_scalar },
    wrt |> ListTensor,
  )
}

pub fn map_tensor_recursively(f: fn(Scalar) -> Scalar, y: Tensor) -> Tensor {
  case y {
    ScalarTensor(s) -> f(s) |> ScalarTensor
    ListTensor(lst) ->
      lst |> list.map(map_tensor_recursively(f, _)) |> ListTensor
  }
}

fn gradient_sigma(y, sigma) {
  case y {
    ScalarTensor(s) -> s.link(s, 1.0, sigma)
    ListTensor(lst) -> gradient_sigma_list(lst, sigma)
  }
}

fn gradient_sigma_list(lst: List(Tensor), sigma) {
  case lst {
    [] -> sigma
    [head, ..rest] -> {
      sigma |> gradient_sigma(head, _) |> gradient_sigma_list(rest, _)
    }
  }
}

//----------------------------
// Differentiable extended numerical functions
//----------------------------

pub fn add_0_0() {
  prim2(float.add, fn(_a, _b, z) { [z, z] })
}

pub fn minus_0_0() {
  prim2(float.subtract, fn(_a, _b, z) { [z, float.negate(z)] })
}

pub fn multiply_0_0() {
  prim2(float.multiply, fn(a, b, z) { [b *. z, a *. z] })
}

pub fn divide_0_0() {
  prim2(fn(x, y) { x /. y }, fn(a, b, z) {
    [z *. { 1.0 /. b }, z *. float.negate(a) /. { b *. b }]
  })
}

pub fn expt_0_0() {
  prim2(
    fn(x, y) {
      let assert Ok(r) = float.power(x, y)
      r
    },
    fn(a, b, z) {
      let assert Ok(r1) = float.power(a, b -. 1.0)
      let assert Ok(r2) = float.power(a, b)
      let assert Ok(log_r) = natural_logarithm(a)
      [z *. b *. r1, z *. r2 *. log_r]
    },
  )
}

/// natural exponential
pub fn exp_0() {
  prim1(exponential, fn(a, z) { z *. exponential(a) })
}

/// natural logarithm
pub fn log_0() {
  prim1(
    fn(x) {
      let assert Ok(r) = natural_logarithm(x)
      r
    },
    fn(a, z) { z *. { 1.0 /. a } },
  )
}

pub fn abs_0() {
  prim1(float.absolute_value, fn(x, z) {
    case x <. 0.0 {
      True -> float.negate(z)
      _ -> z
    }
  })
}

pub fn sqrt_0() {
  prim1(
    fn(x) {
      let assert Ok(r) = float.square_root(x)
      r
    },
    fn(x, z) {
      let assert Ok(r) = float.square_root(x)
      z /. { 2.0 *. r }
    },
  )
}

//------------------------------------
// Differentiable extended tensor functions
//------------------------------------
pub fn tensor_multiply(a, b) {
  let f = multiply_0_0() |> ext2(0, 0)
  f(a, b)
}

pub fn tensor_add(a, b) {
  let f = add_0_0() |> ext2(0, 0)
  f(a, b)
}

pub fn tensor_minus(a, b) {
  let f = minus_0_0() |> ext2(0, 0)
  f(a, b)
}

pub fn tensor_divide(a, b) {
  let f = divide_0_0() |> ext2(0, 0)
  f(a, b)
}

pub fn tensor_expt(a, b) {
  let f = expt_0_0() |> ext2(0, 0)
  f(a, b)
}

pub fn tensor_exp(a) {
  let f = exp_0() |> ext1(0)
  f(a)
}

pub fn tensor_log(a) {
  let f = log_0() |> ext1(0)
  f(a)
}

pub fn tensor_abs(a) {
  let f = abs_0() |> ext1(0)
  f(a)
}

pub fn tensor_sqrt(a) {
  let f = sqrt_0() |> ext1(0)
  f(a)
}

pub fn tensor_sqr(x) {
  tensor_multiply(x, x)
}

//------------------------------------
// sum functions
//------------------------------------
fn sum_1(t) {
  summed(t, to_tensor(0.0))
}

fn summed(t, a) {
  let assert ListTensor(lst) = t
  case lst {
    [] -> a
    [head, ..rest] -> summed(ListTensor(rest), tensor_add(head, a))
  }
}

pub fn tensor_sum(t) {
  let f = sum_1 |> ext1(1)
  f(t)
}

pub fn tensor_sum_cols(t) {
  let f = sum_1 |> ext1(2)
  f(t)
}

//------------------------------------
// star-2-1 operator
//------------------------------------
fn tensor_multiply_1_1(t0, t1) {
  let f = tensor_multiply |> ext2(1, 1)
  f(t0, t1)
}

/// When t0 is a tensor of shape (list m n) and t1 is a tensor of shape (list n),
/// returns a tensor r of shape (list m n) where the m elements of r are formed by
/// multiplying each element of t0 with t1 using the extended multiplication
/// function *.
pub fn tensor_multiply_2_1(t0, t1) {
  let f = tensor_multiply_1_1 |> ext2(2, 1)
  f(t0, t1)
}

//------------------------------------
// Bool Comparasion functions
//------------------------------------
pub fn greater_0_0(a: Scalar, b: Scalar) -> Bool {
  get_real(a) >=. get_real(b)
}

pub fn less_0_0(a, b) {
  get_real(a) <=. get_real(b)
}

pub fn equal_0_0(a, b) {
  get_real(a) == get_real(b)
}

fn lift_scalar_comparator(
  comparator: fn(Scalar, Scalar) -> Bool,
) -> fn(Tensor, Tensor) -> Bool {
  fn(t: Tensor, u: Tensor) {
    case t, u {
      ScalarTensor(s_t), ScalarTensor(s_u) -> comparator(s_t, s_u)
      _, _ -> panic
    }
  }
}

//------------------------------------
// Bool Comparasion functions
//------------------------------------
fn tensorized_comparators(comparator) {
  fn(a: Scalar, b: Scalar) {
    case comparator(a, b) {
      True -> new_scalar(1.0)
      False -> new_scalar(0.0)
    }
  }
  |> lift_scalar2
  |> ext2(0, 0)
}

pub fn tensorized_cmp_greater(a, b) {
  { greater_0_0 |> tensorized_comparators }(a, b)
}

pub fn tensorized_cmp_less(a, b) {
  { less_0_0 |> tensorized_comparators }(a, b)
}

pub fn tensorized_cmp_equal(a, b) {
  { equal_0_0 |> tensorized_comparators }(a, b)
}

//------------------------------------
// argmax
//------------------------------------
pub fn argmax_1(t) {
  let assert ListTensor([head, ..] as lst) = t
  lst
  |> list.index_fold(#(head, 0), fn(acc, item, index) {
    case { greater_0_0 |> lift_scalar_comparator }(item, acc.0) {
      True -> #(item, index)
      _ -> acc
    }
  })
  |> pair.second
  |> int.to_float
  |> new_scalar
  |> ScalarTensor
}

pub fn tensor_argmax(t) {
  let f = argmax_1 |> ext1(1)
  f(t)
}

//------------------------------------
// max
//------------------------------------
pub fn max_1(t) {
  let assert ListTensor([head, ..] as lst) = t
  lst
  |> list.fold(head, fn(acc, item) {
    case { greater_0_0 |> lift_scalar_comparator }(item, acc) {
      True -> item
      _ -> acc
    }
  })
}

pub fn tensor_max(t) {
  let f = max_1 |> ext1(1)
  f(t)
}

//------------------------------------
// correlate
//------------------------------------
pub fn dot_product(t, u) {
  dotted_product(t, u, to_tensor(0.0))
}

pub fn dotted_product(t, u, acc) {
  case t, u {
    ListTensor(lst_t), ListTensor(lst_u) ->
      case lst_t, lst_u {
        [], _ | _, [] -> acc
        [t_head, ..t_rest], [u_head, ..u_rest] -> {
          dotted_product(
            t_rest |> ListTensor,
            u_rest |> ListTensor,
            multiply_0_0()(t_head, u_head) |> add_0_0()(acc),
          )
        }
      }
    _, _ -> panic as "t u have different shapes"
  }
}

pub fn sum_dp(filter: Tensor, signal: Tensor, from: Int, acc: Float) -> Scalar {
  let assert ListTensor(filter_lst) = filter
  let assert ListTensor(signal_lst) = signal

  let filter_length = list.length(filter_lst)

  let zero_tensors = fn() {
    let assert Ok(last_shape) = shape(signal) |> list.last
    build_tensor([last_shape], fn(_idx) { 0.0 })
  }

  let slide_window = fn(lst) {
    case from <= 0, from + filter_length >= list.length(lst) {
      False, False -> lst |> list.drop(from) |> list.take(filter_length)
      True, False ->
        list.repeat(zero_tensors(), times: int.absolute_value(from))
        |> list.append(list.take(lst, from + filter_length))
      False, True ->
        list.drop(lst, from)
        |> list.append(list.repeat(
          zero_tensors(),
          times: int.absolute_value(from),
        ))
      True, True ->
        list.repeat(0.0 |> to_tensor, times: int.absolute_value(from))
        |> list.append(lst)
        |> list.append(list.repeat(
          zero_tensors(),
          times: from + filter_length - list.length(lst),
        ))
    }
  }

  sum_dp_helper(filter_lst, signal_lst |> slide_window, acc |> new_scalar)
}

fn sum_dp_helper(
  filter_lst: List(Tensor),
  signal_lst: List(Tensor),
  acc: Scalar,
) -> Scalar {
  case filter_lst, signal_lst {
    [], _ | _, [] -> acc
    [f_head, ..f_tail], [s_head, ..s_tail] ->
      sum_dp_helper(
        f_tail,
        s_tail,
        dot_product(f_head, s_head)
          |> add_0_0()(acc |> ScalarTensor)
          |> get_scalar,
      )
  }
}

pub fn correlation_overlap(filter, signal, segment) {
  let q = { tlen(filter) - 1 } / 2
  let from = segment - q
  sum_dp(filter, signal, from, 0.0)
}

/// bank: tensor of rank 3, signal tensor of rank 2
pub fn correlate_3_2(bank, signal) {
  build_tensor_from_tensors([signal, bank], fn(items) {
    let assert [signal_item, bank_item] = items
    let segment = signal_item.0
    let filter = bank_item.1
    correlation_overlap(filter, signal, segment) |> ScalarTensor
  })
}

pub fn tensor_correlate(bank, signal) {
  let f = correlate_3_2 |> ext2(3, 2)
  f(bank, signal)
}

//------------------------------------
// Gradient Descent Functions and Hyperparameters
//------------------------------------
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

pub fn revise(f: fn(Theta) -> Theta, revs: Int, theta: Theta) -> Theta {
  case revs {
    0 -> theta
    _ -> revise(f, revs - 1, f(theta))
  }
}

pub fn gradient_descent(hp: Hyperparameters) {
  fn(inflate, deflate, update) {
    fn(obj, theta: Theta) -> Theta {
      let tensor_update = hp |> update

      let f = fn(big_theta: Theta) {
        list.map2(
          big_theta,
          gradient_of(obj, list.map(big_theta, deflate)),
          tensor_update,
        )
      }

      list.map(theta, inflate)
      |> revise(f, hp.revs, _)
      |> list.map(deflate)
    }
  }
}

//------------------------------------
// non-differentiable versions of tensor operations
//------------------------------------
fn lift_numerical_tensor_op(op) {
  fn(t) {
    let f = op |> lift_float1 |> lift_scalar1 |> ext1(0)
    f(t)
  }
}

fn lift_numerical_tensor_op2(op) {
  fn(t, u) {
    let f = op |> lift_float2 |> lift_scalar2 |> ext2(0, 0)
    f(t, u)
  }
}

pub fn numerical_tensor_add(t, u) {
  { float.add |> lift_numerical_tensor_op2 }(t, u)
}

pub fn numerical_tensor_minus(t, u) {
  { float.subtract |> lift_numerical_tensor_op2 }(t, u)
}

pub fn numerical_tensor_multiply(t, u) {
  { float.multiply |> lift_numerical_tensor_op2 }(t, u)
}

pub fn numerical_tensor_divide(t, u) {
  { fn(x, y) { x /. y } |> lift_numerical_tensor_op2 }(t, u)
}

pub fn numerical_tensor_sqr(t) {
  numerical_tensor_multiply(t, t)
}

pub fn numerical_tensor_sqrt(t) {
  {
    fn(x) {
      let assert Ok(r) = float.square_root(x)
      r
    }
    |> lift_numerical_tensor_op
  }(t)
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

// in update function, when computing tensor, automatic differentiation is not required.
// use non-differentiable versions of the tensor functions for better performance
fn naked_u(hp: Hyperparameters) {
  fn(p, g) {
    // p -. hp.alpha *. g
    numerical_tensor_minus(
      p,
      numerical_tensor_multiply(hp.alpha |> to_tensor, g),
    )
  }
}

pub fn naked_gradient_descent(hp: Hyperparameters) {
  { hp |> gradient_descent }(naked_i, naked_d, naked_u)
}

//------------------------------------
// G-velocity
//------------------------------------
pub fn zeros(x) {
  let f = fn(_) { 0.0 |> to_tensor } |> ext1(0)
  f(x)
}

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
    let assert ListTensor([p0, p1]) = pa

    // μ * p1 - α * g
    let v =
      numerical_tensor_minus(
        numerical_tensor_multiply(hp.mu |> to_tensor, p1),
        numerical_tensor_multiply(hp.alpha |> to_tensor, g),
      )
    //[p0 +. v, v]
    [numerical_tensor_add(p0, v), v] |> ListTensor
  }
}

pub fn velocity_gradient_descent(hp: Hyperparameters) {
  { hp |> gradient_descent }(velocity_i, velocity_d, velocity_u)
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

pub fn smooth(decay_rate, average, g) {
  // decay_rate *. average +. { 1.0 -. decay_rate } *. g
  numerical_tensor_add(
    numerical_tensor_multiply(decay_rate, average),
    numerical_tensor_minus(to_tensor(1.0), decay_rate)
      |> numerical_tensor_multiply(g),
  )
}

const epsilon = 1.0e-8

fn rms_u(hp: Hyperparameters) {
  fn(pa, g) {
    let assert ListTensor([p0, p1]) = pa

    let r = smooth(hp.beta |> to_tensor, p1, numerical_tensor_sqr(g))
    let r_sqrt = numerical_tensor_sqrt(r)

    // α / (r_sqrt + ϵ)
    let alpha_hat =
      numerical_tensor_divide(
        hp.alpha |> to_tensor,
        numerical_tensor_add(r_sqrt, epsilon |> to_tensor),
      )
    // [p0 - (α / (r_sqrt + ϵ)) * g, r]
    [numerical_tensor_minus(p0, numerical_tensor_multiply(alpha_hat, g)), r]
    |> ListTensor
  }
}

pub fn rms_gradient_descent(hp: Hyperparameters) {
  { hp |> gradient_descent }(rms_i, rms_d, rms_u)
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
    let assert ListTensor([p0, p1, p2]) = pa

    let r = smooth(hp.beta |> to_tensor, p2, numerical_tensor_sqr(g))
    let r_sqrt = numerical_tensor_sqrt(r)
    let alpha_hat =
      numerical_tensor_divide(
        hp.alpha |> to_tensor,
        numerical_tensor_add(r_sqrt, epsilon |> to_tensor),
      )

    let v = smooth(hp.mu |> to_tensor, p1, g)
    // The accompaniment v is known as the gradient's 1st moment.
    // The accompaniment r is known as the gradient's 2nd moment.
    // p0 - (α / (r_sqrt + ϵ)) * v
    [numerical_tensor_minus(p0, numerical_tensor_multiply(alpha_hat, v)), v, r]
    |> ListTensor
  }
}

pub fn adam_gradient_descent(hp: Hyperparameters) {
  { hp |> gradient_descent }(adam_i, adam_d, adam_u)
}

//------------------------------------
// J-stochastic
//------------------------------------
pub fn samples(n: Int, size: Int) -> List(Int) {
  list.range(0, n - 1) |> list.shuffle |> list.take(size)
}

pub fn sampling_obj(batch_size: Int) {
  fn(expectant, xs, ys) {
    fn(theta) {
      let assert [n, ..] = shape(xs)
      let sample_indices = samples(n, batch_size)

      let sampled_xs = trefs(xs, sample_indices)
      let sampled_ys = trefs(ys, sample_indices)

      theta |> expectant(sampled_xs, sampled_ys)
    }
  }
}

//------------------------------------
// rectify
//------------------------------------
pub fn rectify_0(x: Scalar) {
  case less_0_0(x, new_scalar(0.0)) {
    True -> new_scalar(0.0)
    _ -> x
  }
}

pub fn rectify(t: Tensor) {
  let f = rectify_0 |> lift_scalar1 |> ext1(0)
  f(t)
}

//------------------------------------
// Layer functions
//------------------------------------
pub fn make_theta(weights: dynamic.Dynamic, bias: Float) -> Theta {
  [weights |> tensor, bias |> to_tensor]
}

// Single layer functions

pub fn linear(t) {
  fn(theta: Theta) {
    let assert [a, b] = theta
    tensor_add(tensor_dot_product_2_1(a, t), b)
  }
}

pub fn plane(t) {
  fn(theta: Theta) {
    let assert [a, b] = theta
    tensor_add(tensor_dot_product(a, t), b)
  }
}

pub fn softmax(t) {
  fn(_theta: Theta) {
    let z = tensor_minus(t, tensor_max(t))
    let expz = tensor_exp(z)
    tensor_sum(expz) |> tensor_divide(expz, _)
  }
}

pub fn tensor_dot_product(w, t) {
  tensor_multiply(w, t) |> tensor_sum
}

pub fn tensor_dot_product_2_1(w, t) {
  tensor_multiply_2_1(w, t) |> tensor_sum
}

pub fn relu(t) {
  fn(theta: Theta) { theta |> list.take(2) |> linear(t) |> rectify }
}

pub fn corr(t) {
  fn(theta: Theta) {
    let assert [a, b] = theta
    tensor_correlate(a, t) |> tensor_add(b)
  }
}

/// rectified 1D-convolution function
pub fn recu(t) {
  fn(theta: Theta) { theta |> list.take(2) |> corr(t) |> rectify }
}

// Deep layer functions
pub fn k_relu(k: Int) {
  fn(t: Tensor) {
    fn(theta: Theta) {
      case k {
        0 -> t
        _ -> {
          let next_layer = theta |> relu(t) |> k_relu(k - 1)
          theta |> refr(2) |> next_layer
        }
      }
    }
  }
}

pub fn k_recu(k: Int) {
  fn(t: Tensor) {
    fn(theta: Theta) {
      case k {
        0 -> t
        _ -> {
          let next_layer = theta |> recu(t) |> k_recu(k - 1)
          theta |> refr(2) |> next_layer
        }
      }
    }
  }
}

//------------------------------------
// Loss Functions
//------------------------------------
pub type TargetFn =
  fn(Tensor) -> fn(Theta) -> Tensor

/// SSE loss function (Sum of Squared Error Loss)
pub fn l2_loss(target: TargetFn) {
  fn(xs: Tensor, ys: Tensor) {
    fn(theta: Theta) {
      let pred_ys = theta |> target(xs)
      ys |> tensor_minus(pred_ys) |> tensor_sqr |> tensor_sum
    }
  }
}

//------------------------------------
// Building blocks for neural networks
//------------------------------------
pub type Block {
  Block(block_fn: TargetFn, block_shape: List(Shape))
}

pub fn compose_block_fns(fa, fb, j) {
  fn(t: Tensor) {
    fn(theta: Theta) {
      let f = theta |> fa(t) |> fb
      refr(theta, j) |> f
    }
  }
}

pub fn stack2(ba: Block, bb: Block) {
  Block(
    compose_block_fns(ba.block_fn, bb.block_fn, ba.block_shape |> list.length),
    list.concat([ba.block_shape, bb.block_shape]),
  )
}

pub fn stack_blocks(blocks: List(Block)) -> Block {
  let assert Ok(b) = list.reduce(blocks, fn(acc, block) { stack2(acc, block) })
  b
}

//------------------------------------
//  He Initialization
//------------------------------------
/// Tensors of rank 1 are initialized to contain only 0.0.
///
/// Tensors of rank 2 are initialized to random numbers drawn from a normal
/// distribution with a mean of 0.0 and a variance of (/ 2 fan-in) where fan-in is
/// the last member of the shape.
///
/// Tensors of rank 3 are initialized to random numbers drawn from a normal
/// distribution with a mean of 0.0 and a variance of (/ 2 fan-in) where fan-in is
/// the product of the last two members of the shape.
pub fn init_theta(theta_shape: List(Shape)) -> List(Tensor) {
  theta_shape |> list.map(init_shape)
}

pub fn init_shape(shape: Shape) -> Tensor {
  case list.length(shape) {
    1 -> zero_tensors(shape)
    2 -> {
      let assert [_, fan_in] = shape
      random_tensor(0.0, 2.0 /. int.to_float(fan_in), shape)
    }
    3 -> {
      let assert [_, s1, s2] = shape
      let fan_in = s1 * s2
      random_tensor(0.0, 2.0 /. int.to_float(fan_in), shape)
    }
    _ -> panic
  }
}

fn random_tensor(mean: Float, variance: Float, shape: Shape) -> Tensor {
  build_tensor(shape, fn(_tidx) {
    let assert Ok(r) = float.square_root(variance)
    random_normal(mean, r)
  })
}

fn zero_tensors(shape: Shape) {
  build_tensor(shape, fn(_tidx) { 0.0 })
}

//------------------------------------
// Models and Accuracy
//------------------------------------
pub fn model(target: TargetFn, theta: Theta) {
  fn(t) { theta |> target(t) }
}

pub fn accuracy(a_model: fn(Tensor) -> Tensor, xs: Tensor, ys: Tensor) {
  tensorized_cmp_equal(xs |> a_model |> tensor_argmax, ys |> tensor_argmax)
  |> tensor_sum
  |> tensor_divide(tlen(ys) |> int.to_float |> to_tensor)
  |> get_float
}

pub fn grid_search(
  body,
  good_enough: fn(Theta) -> Bool,
  revs revs_options: List(Int),
  alpha alpha_options: List(Float),
  batch_size batch_size: List(Int),
) {
  [
    revs_options |> list.map(dynamic.from),
    alpha_options |> list.map(dynamic.from),
    batch_size |> list.map(dynamic.from),
  ]
  |> cartesian_product
  |> parallel_map.list_pmap(
    fn(hypers) {
      let assert [revs, alpha, batch_size] = hypers
      let assert Ok(ok_revs) = revs |> dynamic.int
      let assert Ok(ok_alpha) = alpha |> dynamic.float
      let assert Ok(ok_batch_size) = batch_size |> dynamic.int

      let hp = hp_new(ok_revs, ok_alpha) |> hp_new_batch_size(ok_batch_size)

      use <- report_context(hp)

      let theta = body(hp)

      case good_enough(theta) {
        True -> Ok(#(hypers, theta))
        False -> Error(Nil)
      }
    },
    parallel_map.WorkerAmount(16),
    60 * 1000,
  )
}

// how can i stop the parallel execution when I find a Ok value?
// under the hood of parallel_map.list_pmap it use erlang process

// fn tensor_to_list(t: Tensor) {
//   case t {
//     ScalarTensor(s) -> s.real |> dynamic.from
//     ListTensor([]) -> dynamic.from([])
//     ListTensor(lst) -> lst |> list.map(tensor_to_list) |> dynamic.from
//   }
// }

pub fn cartesian_product(lists: List(List(a))) -> List(List(a)) {
  case lists {
    [] -> [[]]
    [first, ..rest] -> {
      let sub_product = cartesian_product(rest)
      list.flat_map(first, fn(item) {
        list.map(sub_product, fn(combo) { [item, ..combo] })
      })
    }
  }
}

fn report_context(hp, do) {
  let start = birl.now()
  let r = do()
  let end = birl.now()
  let diff = birl.difference(end, start)

  report_hypers(hp, diff, result.is_ok(r))
  r
}

fn report_hypers(hp: Hyperparameters, diff: duration.Duration, good: Bool) {
  "{} {} Execution time: {} seconds"
  |> mat.format3(
    case good {
      True -> "Good"
      False -> "Bad"
    },
    hp |> string.inspect,
    duration.blur_to(diff, duration.Second) |> int.to_string,
  )
  |> io.println_error
}
