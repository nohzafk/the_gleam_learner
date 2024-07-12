import gleam/bool
import gleam/dict.{type Dict}
import gleam/dynamic.{type Dynamic}
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/pair
import gleam/result
import gleam/string
import gleam_community/maths/elementary.{exponential, natural_logarithm}
import gluid
import mat

pub fn uuid() {
  gluid.guidv4()
}

//----------------------------
// Real part of a dual is always a tensor (of any rank)
//----------------------------
/// represent the Dual structure
pub type Scalar {
  Scalar(id: String, real: Float, link: Link)
}

/// dual*
pub fn from_scalar(s: Scalar) -> Scalar {
  Scalar(s.id, s.real, end_of_chain)
}

/// dual*
pub fn from_float(v: Float) -> Scalar {
  Scalar(uuid(), v, end_of_chain)
}

pub fn get_real(s: Scalar) {
  s.real
}

//----------------------------
// Chain rule
//----------------------------
pub type Link =
  fn(Scalar, Float, Dict(Scalar, Float)) -> Dict(Scalar, Float)

pub fn end_of_chain(d: Scalar, z: Float, sigma: Dict(Scalar, Float)) {
  let g = dict.get(sigma, d) |> result.unwrap(0.0)
  dict.insert(sigma, d, z +. g)
}

//—————————————————–
// Tensor Basic
// Representation of tensors as nested list of Scalar
//—————————————————–
pub type Tensor {
  ScalarTensor(Scalar)
  ListTensor(List(Tensor))
}

pub type Shape =
  List(Int)

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
        f(new_idx) |> from_float |> ScalarTensor
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

/// Tensor length
pub fn tlen(t: Tensor) -> Int {
  let assert ListTensor(lst) = t
  list.length(lst)
}

/// convert Tensor to nested lis for debug
pub fn tensor_to_list(tensor: Tensor) -> Dynamic {
  case tensor {
    ScalarTensor(scalar) -> dynamic.from(scalar.real)
    ListTensor(tensors) ->
      case tensors {
        [] -> dynamic.from([])
        _ ->
          tensors
          |> list.map(tensor_to_list)
          |> dynamic.from
      }
  }
}

pub fn tensor(lst: Dynamic) -> Tensor {
  case dynamic.classify(lst) {
    "Int" -> {
      use x <- result.map(lst |> dynamic.int)
      x |> int.to_float |> from_float |> ScalarTensor
    }
    "Float" -> {
      use x <- result.map(lst |> dynamic.float)
      from_float(x) |> ScalarTensor
    }
    "List" -> {
      use elements <- result.try(lst |> dynamic.list(dynamic.dynamic))
      let tensors = elements |> list.map(tensor)
      tensors |> ListTensor |> Ok
    }
    type_ -> Error([dynamic.DecodeError("float or list", type_, [])])
  }
  |> result.lazy_unwrap(fn() { panic as "Fail to decode as a Tensor" })
}

//—————————————————–
// Tensor Operators
//—————————————————–
pub fn shape(t: Tensor) -> Shape {
  case t {
    ScalarTensor(_) -> []
    ListTensor([head, ..]) -> {
      [tlen(t), ..shape(head)]
    }
    _ -> panic as "Empty ListTensor"
  }
}

pub fn rank(t: Tensor) -> Int {
  t |> shape |> list.length
}

pub fn size_of(s: Shape) -> Int {
  list.fold(s, 1, fn(acc, i) { acc * i })
}

//—————————————————–
// extend operator for tensor
//—————————————————–

/// Scalar value is considered as rank 0 tensor
pub fn of_rank(n: Int, t: Tensor) {
  case n, t {
    0, ScalarTensor(_) -> True
    _, ScalarTensor(_) -> False
    _, ListTensor([head, ..]) -> of_rank(n - 1, head)
    _, ListTensor([]) -> panic as "Empty ListTensor"
  }
}

pub fn of_ranks(n: Int, t: Tensor, m: Int, u: Tensor) {
  bool.and(of_rank(n, t), of_rank(m, u))
}

pub fn tensor1_map(t: Tensor, f) -> Tensor {
  let assert ListTensor(lst) = t
  lst |> list.map(f) |> ListTensor
}

pub fn tensor2_map(t: Tensor, u: Tensor, f) -> Tensor {
  let assert ListTensor(lst_t) = t
  let assert ListTensor(lst_u) = u
  list.map2(lst_t, lst_u, f) |> ListTensor
}

/// apply f to a tensor t, f is a unary operator on rank n Tensor
pub fn ext1(f, n: Int) {
  fn(t: Tensor) {
    case of_rank(n, t) {
      True -> f(t)
      _ -> t |> tensor1_map(ext1(f, n))
    }
  }
}

/// apply f to 2 tensors t u, f is a binary operator on 2 tensors of rank n, m
pub fn ext2(f, n: Int, m: Int) {
  fn(t: Tensor, u: Tensor) {
    case of_ranks(n, t, m, u) {
      True -> f(t, u)
      _ -> desc(ext2(f, n, m), n, t, m, u)
    }
  }
}

// descend operation g into tensors t of rank n, u of rank m
pub fn desc(g, n: Int, t: Tensor, m: Int, u: Tensor) {
  case rank(t), rank(u) {
    a, _ if a == n -> desc_u(g, t, u)
    _, b if b == m -> desc_t(g, t, u)
    a, b if a == b -> tensor2_map(t, u, g)
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
  t |> tensor1_map(fn(e) { g(e, u) })
}

pub fn desc_u(g, t: Tensor, u: Tensor) {
  u |> tensor1_map(fn(e) { g(t, e) })
}

//----------------------------
// Reverse-mode Auto Differentiation
//----------------------------
pub fn gradient_operator(f, theta: Tensor) -> Tensor {
  let wrt = map_to_scalar(from_scalar, theta)
  gradient_once(f(wrt), wrt)
}

pub fn map_to_scalar(f: fn(Scalar) -> Scalar, y: Tensor) -> Tensor {
  case y {
    ScalarTensor(s) -> f(s) |> ScalarTensor
    ListTensor(lst) -> lst |> list.map(map_to_scalar(f, _)) |> ListTensor
  }
}

pub fn gradient_once(y, wrt) {
  let sigma = gradient_sigma(y, dict.new())
  map_to_scalar(
    fn(s) { sigma |> dict.get(s) |> result.unwrap(0.0) |> from_float },
    wrt,
  )
}

pub fn gradient_sigma(y, sigma) {
  case y {
    ScalarTensor(s) -> s.link(s, 1.0, sigma)
    ListTensor(lst) -> gradient_sigma_list(lst, sigma)
  }
}

fn gradient_sigma_list(lst: List(Tensor), sigma) {
  case lst {
    [] -> sigma
    [head, ..rest] -> {
      gradient_sigma(head, sigma)
      |> gradient_sigma_list(rest, _)
    }
  }
}

// primitive
pub type ScalarOperator1 =
  fn(Scalar) -> Scalar

pub type ScalarOperator2 =
  fn(Scalar, Scalar) -> Scalar

pub fn prim1(real_fn, gradient_fn) -> ScalarOperator1 {
  fn(scalar: Scalar) {
    Scalar(uuid(), real_fn(scalar.real), fn(_d, z, sigma) {
      sigma
      |> scalar.link(scalar, gradient_fn(scalar.real, z), _)
    })
  }
}

pub fn prim2(real_fn, gradient_fn) -> ScalarOperator2 {
  fn(s1: Scalar, s2: Scalar) {
    Scalar(uuid(), real_fn(s1.real, s2.real), fn(_d, z, sigma) {
      let assert [ga, gb] = gradient_fn(s1.real, s2.real, z)
      sigma
      |> s1.link(s1, ga, _)
      |> s2.link(s2, gb, _)
    })
  }
}

// test-helper
const tolerace = 0.0001

pub fn is_tensor_equal(ta: Tensor, tb: Tensor) -> Bool {
  case ta, tb {
    ScalarTensor(Scalar(real: a, ..)), ScalarTensor(Scalar(real: b, ..)) ->
      float.loosely_equals(a, b, tolerace)
    ListTensor(a), ListTensor(b) ->
      case tlen(ta) == tlen(tb) {
        True ->
          list.map2(a, b, is_tensor_equal)
          |> list.fold(True, fn(acc, item) { acc && item })
        _ -> False
      }
    _, _ -> False
  }
}

//----------------------------
// gradient-aware operators
//----------------------------

//----------------------------
// scalar operators
//----------------------------

pub fn add_0_0() {
  prim2(fn(x, y) { x +. y }, fn(_a, _b, z) { [z, z] })
}

pub fn minus_0_0() {
  prim2(fn(x, y) { x -. y }, fn(_a, _b, z) { [z, float.negate(z)] })
}

pub fn multiply_0_0() {
  prim2(fn(x, y) { x *. y }, fn(a, b, z) { [b *. z, a *. z] })
}

pub fn divide_0_0() {
  prim2(fn(x, y) { x /. y }, fn(a, b, z) {
    [z *. { 1.0 /. b }, z *. { 0.0 -. a } /. { b *. b }]
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
  prim1(exponential, fn(a, z) { z *. { 1.0 /. a } })
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

pub type TensorOperator1 =
  fn(Tensor) -> Tensor

pub type TensorOperator2 =
  fn(Tensor, Tensor) -> Tensor

fn scalar_fn_wrapper1(f: ScalarOperator1) -> TensorOperator1 {
  fn(t: Tensor) {
    let assert ScalarTensor(a) = t
    f(a) |> ScalarTensor
  }
}

fn scalar_fn_wrapper2(f: ScalarOperator2) -> TensorOperator2 {
  fn(ta: Tensor, tb: Tensor) {
    let assert ScalarTensor(a) = ta
    let assert ScalarTensor(b) = tb
    f(a, b) |> ScalarTensor
  }
}

//------------------------------------
// Extended functions.
//------------------------------------
pub fn tensor_multiply(a, b) {
  let f = multiply_0_0() |> scalar_fn_wrapper2 |> ext2(0, 0)
  f(a, b)
}

pub fn tensor_add(a, b) {
  let f = add_0_0() |> scalar_fn_wrapper2 |> ext2(0, 0)
  f(a, b)
}

pub fn tensor_minus(a, b) {
  let f = minus_0_0() |> scalar_fn_wrapper2 |> ext2(0, 0)
  f(a, b)
}

pub fn tensor_divide(a, b) {
  let f = divide_0_0() |> scalar_fn_wrapper2 |> ext2(0, 0)
  f(a, b)
}

pub fn tensor_expt(a, b) {
  let f = expt_0_0() |> scalar_fn_wrapper2 |> ext2(0, 0)
  f(a, b)
}

pub fn tensor_exp(a) {
  let f = exp_0() |> scalar_fn_wrapper1 |> ext1(0)
  f(a)
}

pub fn tensor_log(a) {
  let f = log_0() |> scalar_fn_wrapper1 |> ext1(0)
  f(a)
}

pub fn tensor_abs(a) {
  let f = abs_0() |> scalar_fn_wrapper1 |> ext1(0)
  f(a)
}

pub fn tensor_sqrt(a) {
  let f = sqrt_0() |> scalar_fn_wrapper1 |> ext1(0)
  f(a)
}

pub fn tensor_sqr(x) {
  tensor_multiply(x, x)
}

//------------------------------------
// Comparators
//------------------------------------
fn comparator(f) {
  fn(ta, tb) {
    let assert ScalarTensor(sa) = ta
    let assert ScalarTensor(sb) = tb
    f(sa.real, sb.real)
  }
}

pub fn greater_0_0(a, b) {
  comparator(fn(x, y) { x >. y })(a, b)
}

//------------------------------------
// star-2-1 operator
//------------------------------------
fn tensor_multiply_1_1(a, b) {
  let f = tensor_multiply |> ext2(1, 1)
  f(a, b)
}

pub fn tensor_multiply_2_1(a, b) {
  let f = tensor_multiply_1_1 |> ext2(2, 1)
  f(a, b)
}

//------------------------------------
// sum functions
//------------------------------------
fn sum_1(t) {
  summed(t, 0.0 |> from_float |> ScalarTensor)
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
// argmax
//------------------------------------
pub fn argmax_1(t) {
  let assert ListTensor([head, ..] as lst) = t
  lst
  |> list.index_fold(#(head, 0), fn(acc, item, index) {
    case greater_0_0(item, acc.0) {
      True -> #(item, index)
      _ -> acc
    }
  })
  |> pair.second
  |> int.to_float
  |> from_float
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
    case greater_0_0(item, acc) {
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
  dotted_product(t, u, 0.0 |> from_float)
}

pub fn dotted_product(t, u, acc) {
  case t, u {
    ListTensor(lst_t), ListTensor(lst_u) ->
      case lst_t, lst_u {
        [], _ | _, [] -> acc
        [t_head, ..t_rest], [u_head, ..u_rest] -> {
          let assert ScalarTensor(s_t) = t_head
          let assert ScalarTensor(s_u) = u_head
          dotted_product(
            t_rest |> ListTensor,
            u_rest |> ListTensor,
            multiply_0_0()(s_t, s_u) |> add_0_0()(acc),
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
        list.repeat(
          0.0 |> from_float |> ScalarTensor,
          times: int.absolute_value(from),
        )
        |> list.append(lst)
        |> list.append(list.repeat(
          zero_tensors(),
          times: from + filter_length - list.length(lst),
        ))
    }
  }

  sum_dp_helper(filter_lst, signal_lst |> slide_window, acc |> from_float)
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
        dot_product(f_head, s_head) |> add_0_0()(acc),
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
