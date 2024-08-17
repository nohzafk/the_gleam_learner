///// Flat binary tensor implementation

import gleam/bit_array
import gleam/dict.{type Dict}
import gleam/dynamic.{type Dynamic}
import gleam/erlang.{type Reference}
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/result
import gleam/string

import gleam_community/maths/elementary.{exponential, natural_logarithm}
import mat

//----------------------------
//  Flat binary tensor
//----------------------------
pub const tolerace = 0.0001

pub type Shape =
  List(Int)

pub fn get_id() {
  erlang.make_reference()
}

pub type Tensor {
  FlatTensor(
    id: Reference,
    shape: Shape,
    store: BitArray,
    offset: Int,
    size: Int,
    strides: List(Int),
    rank: Int,
  )
}

/// build a bit array
pub fn build_store(size: Int, f: fn(Int) -> Float) -> BitArray {
  build_store_helper(f, size, 0, <<>>)
}

fn build_store_helper(f: fn(Int) -> Float, size: Int, i: Int, array: BitArray) {
  case i == size {
    True -> array
    _ -> build_store_helper(f, size, i + 1, <<array:bits, f(i):float>>)
  }
}

pub fn new_flat(shape shape: Shape, store store: BitArray, offset offset: Int) {
  FlatTensor(
    id: get_id(),
    shape: shape,
    store: store,
    offset: offset,
    size: size_of(shape),
    strides: strides(shape),
    rank: list.length(shape),
  )
}

pub fn float_to_tensor(v: Float) {
  FlatTensor(
    id: get_id(),
    shape: [],
    store: <<v:float>>,
    offset: 0,
    size: 0,
    strides: [],
    rank: 0,
  )
}

pub fn size_of(shape: Shape) {
  list.fold(shape, 1, fn(acc, i) { acc * i })
}

fn strides(shape: Shape) -> List(Int) {
  case shape {
    [] -> []
    [_head, ..tail] -> [size_of(tail), ..strides(tail)]
  }
}

pub fn tensor_equal(actual: Tensor, expect: Tensor) -> Bool {
  case actual, expect {
    FlatTensor(shape: t_shape, ..), FlatTensor(shape: u_shape, ..)
      if t_shape != u_shape
    -> False
    t, u -> t.size == u.size && equal_elements(t, u)
  }
}

pub fn equal_elements(actual: Tensor, expect: Tensor) -> Bool {
  let actual_offset_size = actual.offset * 64
  let expect_offset_size = expect.offset * 64
  let assert <<_:size(actual_offset_size), actual_bits:bits>> = actual.store
  let assert <<_:size(expect_offset_size), expect_bits:bits>> = expect.store

  equal_bit_array_by_float(actual_bits, expect_bits)
}

fn equal_bit_array_by_float(actual_bits: BitArray, expect_bits: BitArray) {
  case actual_bits, expect_bits {
    <<>>, <<>> -> True
    <<a_float:float, a_tail:bits>>, <<b_float:float, b_tail:bits>> -> {
      float.loosely_equals(a_float, b_float, tolerace)
      && equal_bit_array_by_float(a_tail, b_tail)
    }
    _, _ -> False
  }
}

//----------------------------
//  Tensor basics
//----------------------------

pub fn tref(t: Tensor, i: Int) {
  case t.rank {
    1 -> {
      let offset_bits = { t.offset + i } * 64
      let assert <<_omit:size(offset_bits), v:float, _:bits>> = t.store
      float_to_tensor(v)
    }
    _ -> {
      let assert [stride, ..] = t.strides
      new_flat(t.shape |> list.drop(1), t.store, t.offset + i * stride)
    }
  }
}

pub fn trefs(t: Tensor, b: List(Int)) {
  let est = t.shape |> list.drop(1)
  let estride = size_of(est)

  let final_store =
    b
    |> list.map(fn(position) {
      // the unit of "at" and "take" parameter is byte
      // a float value is 64bits, that's 8 bytes
      let assert Ok(slice) =
        bit_array.slice(
          from: t.store,
          at: position * estride * 8,
          take: estride * 8,
        )
      slice
    })
    |> bit_array.concat

  let new_shape = [list.length(b), ..est]
  new_flat(shape: new_shape, store: final_store, offset: 0)
}

pub fn tlen(t: Tensor) {
  let assert [head, ..] = t.shape
  head
}

// Gleam language
// this function currently accept int, float, list of int, list of float, list of list
pub fn to_tensor(data: Dynamic) -> Tensor {
  case dynamic.classify(data) {
    "Int" -> {
      use x <- result.map(data |> dynamic.int)
      x |> int.to_float |> float_to_tensor
    }
    "Float" -> {
      use x <- result.map(data |> dynamic.float)
      x |> float_to_tensor
    }
    "List" -> {
      use elements <- result.try(data |> dynamic.list(dynamic.dynamic))
      let tensors = elements |> list.map(to_tensor)
      tensors |> merge_flat_tensors |> Ok
    }
    type_ -> {
      Error([
        dynamic.DecodeError(
          "a number or a list of numbers",
          type_ <> ": " <> string.inspect(data),
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

pub fn floats_to_tensor(floats: List(Float)) {
  let store =
    list.fold(floats, <<>>, fn(acc, float) {
      bit_array.append(acc, <<float:float>>)
    })
  new_flat(shape: [list.length(floats)], store: store, offset: 0)
}

pub fn merge_flat_tensors(lst: List(Tensor)) -> Tensor {
  let assert [head, ..] = lst

  let inner_shape = head.shape
  let outer = list.length(lst)
  let new_shape = [outer, ..inner_shape]

  let final_store =
    list.fold(lst, <<>>, fn(acc, arg) { bit_array.append(acc, arg.store) })

  new_flat(shape: new_shape, store: final_store, offset: 0)
}

/// build tensor with shape, f is supplied with indexes
pub fn build_tensor(shape: Shape, f) {
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
        f(new_idx)
      })
      |> floats_to_tensor

    [s, ..rest] ->
      list.range(0, s - 1)
      |> list.map(fn(i) {
        let new_idx = list.append(idx, [i])
        build_tensor_helper(f, rest, new_idx)
      })
      |> merge_flat_tensors

    [] -> panic as "Shape cannot be empty"
  }
}

//----------------------------
//  tensor operations
//----------------------------
pub fn shape(t: Tensor) {
  t.shape
}

pub fn store(t: Tensor) {
  t.store
}

pub fn rank(t: Tensor) {
  t.shape |> list.length
}

pub fn reshape(t: Tensor, shape: Shape) {
  case size_of(shape) == t.size {
    True -> new_flat(shape: shape, store: t.store, offset: t.offset)
    _ -> panic as mat.format2("Cannot reshape {} to {}", t.shape, shape)
  }
}

//—————————————————–—————————————————–—————————————————–
// Pointwise extension helper functions
//—————————————————–—————————————————–—————————————————–

/// try to convert a bitarray to a float value
fn scalarize(bits: BitArray) -> Float {
  let assert <<v:float>> = bits
  v
}

pub fn min_shape(min_rank: Int, in_shape: Shape) -> Shape {
  in_shape
  |> list.drop(list.length(in_shape) - min_rank)
}

/// calculates indices for multi-dimensional array access based on strides. It takes four parameters:
///
/// 1. strides: A list of stride vectors
/// 2. out-i: An initial index
/// 3. i0 and i1: Initial values for two dimensions
///
/// It iterates through the strides, updating i0, i1, and x (initially out-i) at each step.
/// The function returns the final values of i0 and i1.
pub fn idxs(strides: List(#(Int, Int, Int)), out_i, i0, i1) {
  let tuple =
    list.fold(strides, #(i0, i1, out_i), fn(acc, stride) {
      let #(curr_i0, curr_i1, x) = acc
      let idx = x / stride.0
      let next_x = x % stride.0
      #(curr_i0 + idx * stride.1, curr_i1 + idx * stride.2, next_x)
    })

  #(tuple.0, tuple.1)
}

pub fn merge_shapes(in_shape: Shape, min_rank: Int, out_f_shape: Shape) {
  in_shape
  |> list.take(list.length(in_shape) - min_rank)
  |> list.append(out_f_shape)
}

pub fn float_bits_walker(f, slice: BitArray, acc) {
  case slice {
    <<>> -> acc
    <<head:float, tail:bits>> -> float_bits_walker(f, tail, f(acc, head))
    _ -> panic as "Invalid bit array of floats"
  }
}

pub fn bitarray_to_floats(slice: BitArray) {
  float_bits_walker(fn(acc, i) { [i, ..acc] }, slice, []) |> list.reverse
}

pub fn to_bitarray(floats: List(Float)) -> BitArray {
  floats
  |> list.fold(<<>>, fn(acc, v) { <<acc:bits, v:float>> })
}

pub fn lower_float1(f: fn(Float) -> Float) -> fn(BitArray) -> BitArray {
  fn(a_slice) {
    let assert <<a:float>> = a_slice
    let b = f(a)
    <<b:float>>
  }
}

pub fn lower_float2(
  f: fn(Float, Float) -> Float,
) -> fn(BitArray, BitArray) -> BitArray {
  fn(a_slice, b_slice) {
    let assert <<a:float>> = a_slice
    let assert <<b:float>> = b_slice
    let c = f(a, b)
    <<c:float>>
  }
}

pub fn lower_float3(
  f: fn(Float, Float, Float) -> #(Float, Float),
) -> fn(BitArray, BitArray, BitArray) -> #(BitArray, BitArray) {
  fn(a_slice, b_slice, c_slice) {
    let assert <<a:float>> = a_slice
    let assert <<b:float>> = b_slice
    let assert <<c:float>> = c_slice
    let #(d, e) = f(a, b, c)
    #(<<d:float>>, <<e:float>>)
  }
}

//—————————————————–—————————————————–—————————————————–
// Unary Pointwise extension
//—————————————————–—————————————————–—————————————————–

// element-wise multiplication with broadcasting
pub fn extend_rank1_numeric(f, m, shape_fn) {
  fn(t: Tensor) -> Tensor {
    case t.rank {
      0 -> f(t.store) |> scalarize |> float_to_tensor
      _ -> flat_extend_rank1_numeric(f, m, shape_fn, t)
    }
  }
}

pub fn flat_extend_rank1_numeric(
  f: fn(BitArray) -> BitArray,
  min_rank: Int,
  shape_fn: fn(Shape) -> Shape,
  t0: Tensor,
) -> Tensor {
  let s0 = t0.shape
  let sf0 = min_shape(min_rank, s0)
  // how many float
  let stride0 = size_of(sf0)

  let v_out =
    list.range(0, t0.size / stride0 - 1)
    |> list.map(fn(i0) {
      // slice use bytes as unit, one float value is 64 bits which is 8 bytes
      let assert Ok(slice) =
        t0.store |> bit_array.slice(t0.offset + i0 * stride0 * 8, stride0 * 8)
      f(slice)
    })
    |> bit_array.concat

  let s_out = merge_shapes(s0, min_rank, shape_fn(sf0))

  new_flat(shape: s_out, store: v_out, offset: 0)
}

pub fn extend_rank1_gradient(f, m, shape_fn) {
  fn(t: Tensor, z: Tensor) -> Tensor {
    case t.rank {
      0 -> f(t.store, z.store) |> scalarize |> float_to_tensor
      _ -> flat_extend_rank1_gradient(f, m, shape_fn, t, z)
    }
  }
}

pub fn flat_extend_rank1_gradient(
  tensor_store_f: fn(BitArray, BitArray) -> BitArray,
  min_rank: Int,
  shape_fn: fn(Shape) -> Shape,
  t0: Tensor,
  z: Tensor,
) {
  // z has the same shape as the output
  let s0 = t0.shape
  let sf0 = min_shape(min_rank, s0)
  let stride0 = size_of(sf0)
  let stride_z = size_of(shape_fn(sf0))

  let v_out =
    list.range(0, t0.size / stride0 - 1)
    |> list.map(fn(i0) {
      let assert Ok(t0_slice) =
        t0.store |> bit_array.slice(t0.offset + i0 * stride0 * 8, stride0 * 8)
      let assert Ok(z_slice) =
        z.store |> bit_array.slice(z.offset + i0 * stride_z * 8, stride_z * 8)
      tensor_store_f(t0_slice, z_slice)
    })
    |> bit_array.concat

  new_flat(shape: s0, store: v_out, offset: 0)
}

//—————————————————–—————————————————–—————————————————–
// Binary Pointwise extension
//—————————————————–—————————————————–—————————————————–

// element-wise multiplication with broadcasting
pub fn extend_rank2_numeric(f, m: Int, n: Int, shape_fn) {
  fn(t: Tensor, u: Tensor) -> Tensor {
    case t.rank, u.rank {
      0, 0 -> f(t.store, u.store) |> scalarize |> float_to_tensor
      _, _ -> flat_extend_rank2_numeric(f, m, n, shape_fn, t, u)
    }
  }
}

pub fn flat_extend_rank2_numeric(
  f: fn(BitArray, BitArray) -> BitArray,
  r0: Int,
  r1: Int,
  shape_fn,
  t0: Tensor,
  t1: Tensor,
) {
  let s0 = t0.shape
  let sf0 = min_shape(r0, s0)
  let stride0 = size_of(sf0)

  let s1 = t1.shape
  let sf1 = min_shape(r1, s1)
  let stride1 = size_of(sf1)

  let sf_out = shape_fn(sf0, sf1)
  let stride_out = size_of(sf_out)

  extend_rank2_shapes(
    s0,
    s1,
    r0,
    r1,
    sf_out,
    fn(s_out, size_out, _q0, _q1, strides) {
      let v_out =
        list.range(0, size_out / stride_out - 1)
        |> list.map(fn(out_i) {
          let #(i0, i1) =
            idxs(strides, out_i * stride_out, t0.offset, t1.offset)

          let assert Ok(t0_slice) =
            t0.store |> bit_array.slice(i0 * 8, stride0 * 8)
          let assert Ok(t1_slice) =
            t1.store |> bit_array.slice(i1 * 8, stride1 * 8)
          f(t0_slice, t1_slice)
        })
        |> bit_array.concat

      new_flat(shape: s_out, store: v_out, offset: 0)
    },
  )
}

pub fn extend_rank2_shapes(
  s0: Shape,
  s1: Shape,
  r0: Int,
  r1: Int,
  sf_out: Shape,
  k,
) {
  let l0 = list.length(s0)
  let l1 = list.length(s1)
  case s0, s1 {
    _, _ if r0 == l0 && r1 == l1 ->
      k(sf_out, size_of(sf_out), size_of(s0), size_of(s1), [])

    _, [s1_head, ..s1_tail] if r0 == l0 ->
      extend_rank2_shapes(s0, s1_tail, r0, r1, sf_out, desc_right(s1_head, k))

    [s0_head, ..s0_tail], _ if r1 == l1 ->
      extend_rank2_shapes(s0_tail, s1, r0, r1, sf_out, desc_left(s0_head, k))

    [s0_head, ..s0_tail], [s1_head, ..s1_tail]
      if l0 > 0 && l1 > 0 && s0_head == s1_head
    ->
      extend_rank2_shapes(
        s0_tail,
        s1_tail,
        r0,
        r1,
        sf_out,
        desc_both(s0_head, k),
      )

    _, [s1_head, ..s1_tail] if l1 > l0 ->
      extend_rank2_shapes(s0, s1_tail, r0, r1, sf_out, desc_right(s1_head, k))

    [s0_head, ..s0_tail], _ if l0 > l1 ->
      extend_rank2_shapes(s0_tail, s1, r0, r1, sf_out, desc_left(s0_head, k))
    _, _ ->
      panic as mat.format4(
        "Shapes are incompatible for ext2: {}, and {} for min ranks {}, and {}",
        s0,
        s1,
        r0,
        r1,
      )
  }
}

pub fn desc_both(d: Int, k) {
  fn(s_out: Shape, qout: Int, q0: Int, q1: Int, strides) {
    k(
      //
      [d, ..s_out],
      //
      qout * d,
      //
      q0 * d,
      //
      q1 * d,
      //
      [#(qout, q0, q1), ..strides],
    )
  }
}

pub fn desc_left(d: Int, k) {
  fn(s_out: Shape, qout: Int, q0: Int, q1: Int, strides) {
    k(
      //
      [d, ..s_out],
      //
      qout * d,
      //
      q0 * d,
      //
      q1,
      //
      [#(qout, q0, 0), ..strides],
    )
  }
}

pub fn desc_right(d: Int, k) {
  fn(s_out: Shape, qout: Int, q0: Int, q1: Int, strides) {
    k(
      //
      [d, ..s_out],
      //
      qout * d,
      //
      q0,
      //
      q1 * d,
      //
      [#(qout, 0, q1), ..strides],
    )
  }
}

pub fn extend_rank2_gradient(f, m, n, shape_fn) {
  fn(t: Tensor, u: Tensor, z: Tensor) -> #(Tensor, Tensor) {
    case t.rank, u.rank {
      0, 0 ->
        f(t.store, u.store, z.store)
        |> fn(stores: #(BitArray, BitArray)) {
          #(
            stores.0 |> scalarize |> float_to_tensor,
            stores.1 |> scalarize |> float_to_tensor,
          )
        }
      _, _ -> flat_extend_rank2_gradient(f, m, n, shape_fn, t, u, z)
    }
  }
}

pub fn flat_extend_rank2_gradient(
  tensor_store_f: fn(BitArray, BitArray, BitArray) -> #(BitArray, BitArray),
  r0,
  r1,
  shape_fn,
  t0: Tensor,
  t1: Tensor,
  z: Tensor,
) {
  let s0 = t0.shape
  let sf0 = min_shape(r0, s0)
  let stride0 = size_of(sf0)

  let s1 = t1.shape
  let sf1 = min_shape(r1, s1)
  let stride1 = size_of(sf1)

  let sf_z = shape_fn(sf0, sf1)
  let stride_z = size_of(sf_z)

  extend_rank2_shapes(s0, s1, r0, r1, sf_z, fn(_sz, size_z, _q0, _q1, strides) {
    let #(final_g0, final_g1) =
      list.range(0, size_z / stride_z - 1)
      |> list.fold(#(new_vec(size_of(s0)), new_vec(size_of(s1))), fn(acc, iz) {
        let #(i0, i1) = idxs(strides, iz * stride_z, t0.offset, t1.offset)
        let assert Ok(t0_slice) =
          t0.store |> bit_array.slice(i0 * 8, stride0 * 8)
        let assert Ok(t1_slice) =
          t1.store |> bit_array.slice(i1 * 8, stride1 * 8)
        let assert Ok(z_slice) =
          z.store |> bit_array.slice(z.offset + iz * stride_z * 8, stride_z * 8)

        let #(g0_slice, g1_slice) = tensor_store_f(t0_slice, t1_slice, z_slice)

        #(
          accumulate_gradient(acc.0, i0, g0_slice),
          accumulate_gradient(acc.1, i1, g1_slice),
        )
      })

    #(
      new_flat(shape: s0, store: final_g0, offset: 0),
      new_flat(shape: s1, store: final_g1, offset: 0),
    )
  })
}

fn accumulate_gradient(acc: BitArray, offset: Int, grad: BitArray) {
  let omit_bits = offset * 64
  let assert <<before:size(omit_bits), acc_float:float, after:bits>> = acc
  let assert <<grad_float:float>> = grad

  let new_float = acc_float +. grad_float

  <<before:size(omit_bits), new_float:float, after:bits>>
  // <<before, new_float:float, after:bits>> is a valid expression, but wrong result
}

fn new_vec(size: Int) -> BitArray {
  list.repeat(<<0.0:float>>, size) |> bit_array.concat
}

//—————————————————–—————————————————–—————————————————–
// Automaic Differentiation
//—————————————————–—————————————————–—————————————————–
pub type Dual {
  Dual(id: Reference, tensor: Tensor, link: Link)
}

pub fn to_dual(t: Tensor) -> Dual {
  Dual(id: get_id(), tensor: t, link: end_of_chain)
}

pub fn from_dual(d: Dual) -> Dual {
  Dual(id: d.id, tensor: d.tensor, link: end_of_chain)
}

pub type GradientState =
  Dict(Dual, Tensor)

pub type Link =
  fn(Dual, Tensor, GradientState) -> GradientState

/// Chain rule
/// non-differentiable add
pub fn add_numeric(a, b) {
  let f =
    float.add |> lower_float2 |> extend_rank2_numeric(0, 0, fn(_, _) { [] })
  f(a, b)
}

pub fn end_of_chain(d: Dual, z: Tensor, sigma: GradientState) {
  let g = dict.get(sigma, d) |> result.unwrap(float_to_tensor(0.0))
  dict.insert(sigma, d, add_numeric(z, g))
}

//----------------------------
// Reverse-mode AD
//----------------------------
pub type Differentiable {
  DualDiff(Dual)
  ListDiff(List(Differentiable))
}

pub fn to_diff(data: Dynamic) -> Differentiable {
  case dynamic.classify(data) {
    "Int" -> {
      use _ <- result.map(data |> dynamic.int)
      data |> to_tensor |> to_dual |> DualDiff
    }
    "Float" -> {
      use _ <- result.map(data |> dynamic.float)
      data |> to_tensor |> to_dual |> DualDiff
    }
    "List" -> {
      use elements <- result.map(data |> dynamic.list(dynamic.dynamic))
      elements
      |> list.map(fn(ele) {
        ele |> dynamic.from |> to_tensor |> to_dual |> DualDiff
      })
      |> ListDiff
    }
    type_ -> {
      Error([
        dynamic.DecodeError(
          "a number or a list of numbers",
          mat.format2("{}: {}", type_, data),
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

pub fn map_tensor_recursively(
  f: fn(Dual) -> Dual,
  y: Differentiable,
) -> Differentiable {
  case y {
    DualDiff(d) -> f(d) |> DualDiff
    ListDiff(lst) -> lst |> list.map(map_tensor_recursively(f, _)) |> ListDiff
  }
}

pub fn gradient_of(
  f: fn(Differentiable) -> Differentiable,
  theta: Differentiable,
) {
  let wrt = map_tensor_recursively(from_dual, theta)
  gradient_once(f(wrt), wrt)
}

pub fn gradient_once(y, wrt) {
  let sigma = gradient_sigma(y, dict.new())
  map_tensor_recursively(
    fn(d) {
      sigma |> dict.get(d) |> result.unwrap(float_to_tensor(0.0)) |> to_dual
    },
    wrt,
  )
}

pub fn one_like(t: Tensor) {
  new_flat(t.shape, list.repeat(1.0, size_of(t.shape)) |> to_bitarray, 0)
}

fn gradient_sigma(y, sigma) {
  case y {
    DualDiff(d) -> d.link(d, one_like(d.tensor), sigma)
    ListDiff(lst) -> gradient_sigma_list(lst, sigma)
  }
}

fn gradient_sigma_list(lst: List(Differentiable), sigma) {
  case lst {
    [] -> sigma
    [head, ..rest] -> {
      sigma |> gradient_sigma(head, _) |> gradient_sigma_list(rest, _)
    }
  }
}

//----------------------------
// Dualized tensor op creators
//----------------------------
pub type Prim1Fn {
  Prim1Fn(
    numeric_fn: fn(Float) -> Float,
    gradient_fn: fn(Float, Float) -> Float,
    shape_fn: fn(Shape) -> Shape,
  )
}

pub type Prim2Fn {
  Prim2Fn(
    numeric_fn: fn(Float, Float) -> Float,
    gradient_fn: fn(Float, Float, Float) -> #(Float, Float),
    shape_fn: fn(Shape, Shape) -> Shape,
  )
}

pub fn default_shape_fn_1(shape: Shape) -> Shape {
  case shape {
    [] -> []
    [head, ..] -> [head]
  }
}

pub fn default_shape_fn_2(shape: Shape, _: Shape) -> Shape {
  shape
}

/// prim_dual function creates a differentiable function that operates on Dual values
///
/// Constructs a differentiable function (known as a primitive) of one Dual value
/// argument that invokes ρ-fn to compute the result of the application of the
/// primitive, and uses ∇-fn to find the gradient of the result with respect to the
/// argument provided to the primitive.
pub fn prim1_dual(numeric_fn, gradient_fn) {
  fn(da: Dual) {
    Dual(
      //
      id: get_id(),
      //
      tensor: numeric_fn(da.tensor),
      //
      link: fn(_d, z, sigma) {
        let ga = gradient_fn(da.tensor, z)
        sigma
        |> da.link(da, ga, _)
      },
    )
  }
}

pub fn prim2_dual(numeric_fn, gradient_fn) {
  fn(da: Dual, db: Dual) {
    Dual(
      //
      id: get_id(),
      //
      tensor: numeric_fn(da.tensor, db.tensor),
      //
      link: fn(_d, z, sigma) {
        let #(ga, gb) = gradient_fn(da.tensor, db.tensor, z)
        sigma
        |> da.link(da, ga, _)
        |> db.link(db, gb, _)
      },
    )
  }
}

/// ext function extend the prim opeartor to apply to tensors of certain shapes
pub fn ext1(prim_fn: Prim1Fn, n: Int) {
  prim1_dual(
    extend_rank1_numeric(
      prim_fn.numeric_fn |> lower_float1,
      n,
      prim_fn.shape_fn,
    ),
    extend_rank1_gradient(
      prim_fn.gradient_fn |> lower_float2,
      n,
      prim_fn.shape_fn,
    ),
  )
}

pub fn ext2(prim_fn: Prim2Fn, m: Int, n: Int) {
  prim2_dual(
    extend_rank2_numeric(
      prim_fn.numeric_fn |> lower_float2,
      m,
      n,
      prim_fn.shape_fn,
    ),
    extend_rank2_gradient(
      prim_fn.gradient_fn |> lower_float3,
      m,
      n,
      prim_fn.shape_fn,
    ),
  )
}

//----------------------------
// A-scalar-ops
//----------------------------
pub fn unwrap_ok_number(op, x) {
  let assert Ok(r) = op(x)
  r
}

pub fn unwrap_ok_number2(op, x, y) {
  let assert Ok(r) = op(x, y)
  r
}

pub fn add_0_0() {
  Prim2Fn(
    numeric_fn: float.add,
    gradient_fn: fn(_a, _b, z) { #(z, z) },
    shape_fn: default_shape_fn_2,
  )
}

pub fn subtract_0_0() {
  Prim2Fn(
    numeric_fn: float.subtract,
    gradient_fn: fn(_a, _b, z) { #(z, float.negate(z)) },
    shape_fn: default_shape_fn_2,
  )
}

pub fn multiply_0_0() {
  Prim2Fn(
    numeric_fn: float.multiply,
    gradient_fn: fn(a, b, z) { #(b *. z, a *. z) },
    shape_fn: default_shape_fn_2,
  )
}

pub fn divide_0_0() {
  Prim2Fn(
    numeric_fn: fn(x, y) { x /. y },
    gradient_fn: fn(a, b, z) {
      #(z *. { 1.0 /. b }, z *. float.negate(a) /. { b *. b })
    },
    shape_fn: default_shape_fn_2,
  )
}

pub fn expt_0_0() {
  Prim2Fn(
    numeric_fn: fn(x, y) {
      let assert Ok(r) = float.power(x, y)
      r
    },
    gradient_fn: fn(a, b, z) {
      let assert Ok(r1) = float.power(a, b -. 1.0)
      let assert Ok(r2) = float.power(a, b)
      let assert Ok(log_r) = natural_logarithm(a)
      #(z *. b *. r1, z *. r2 *. log_r)
    },
    shape_fn: default_shape_fn_2,
  )
}

/// natural exponential
pub fn exp_0() {
  Prim1Fn(
    numeric_fn: exponential,
    gradient_fn: fn(a, z) { z *. exponential(a) },
    shape_fn: default_shape_fn_1,
  )
}

/// natural logarithm
pub fn log_0() {
  Prim1Fn(
    numeric_fn: unwrap_ok_number(natural_logarithm, _),
    gradient_fn: fn(a, z) { z *. { 1.0 /. a } },
    shape_fn: default_shape_fn_1,
  )
}

pub fn sqrt_0() {
  Prim1Fn(
    numeric_fn: unwrap_ok_number(float.square_root, _),
    gradient_fn: fn(x, z) {
      let assert Ok(r) = float.square_root(x)
      z /. { 2.0 *. r }
    },
    shape_fn: default_shape_fn_1,
  )
}

pub fn abs_0() {
  Prim1Fn(
    numeric_fn: float.absolute_value,
    gradient_fn: fn(x, z) {
      case x <. 0.0 {
        True -> float.negate(z)
        _ -> z
      }
    },
    shape_fn: default_shape_fn_1,
  )
}

pub fn rectify_0() {
  Prim1Fn(
    numeric_fn: fn(s) {
      case s <. 0.0 {
        True -> 0.0
        _ -> s
      }
    },
    gradient_fn: fn(s, z) {
      case s <. 0.0 {
        True -> 0.0
        _ -> z
      }
    },
    shape_fn: fn(s) { s },
  )
}

//------------------------------------
// differentiable extended functions.
//------------------------------------

pub fn d_multiply(da, db) {
  { multiply_0_0() |> ext2(0, 0) }(da, db)
}

pub fn d_add(da, db) {
  { add_0_0() |> ext2(0, 0) }(da, db)
}

pub fn d_subtract(da, db) {
  { subtract_0_0() |> ext2(0, 0) }(da, db)
}

pub fn d_divide(da, db) {
  { divide_0_0() |> ext2(0, 0) }(da, db)
}

pub fn d_expt(da, db) {
  { expt_0_0() |> ext2(0, 0) }(da, db)
}

pub fn d_exp(da) {
  { exp_0() |> ext1(0) }(da)
}

pub fn d_log(da) {
  { log_0() |> ext1(0) }(da)
}

pub fn d_abs(da) {
  { abs_0() |> ext1(0) }(da)
}

pub fn d_rectify(da) {
  { rectify_0() |> ext1(0) }(da)
}

pub fn d_sqrt(da) {
  { sqrt_0() |> ext1(0) }(da)
}

pub fn d_sqr(da) {
  d_multiply(da, da)
}

//------------------------------------
// non-differentiable extended functions.
//------------------------------------
pub fn scalar_shape_fn_1(_) {
  []
}

pub fn scalar_shape_fn_2(_, _) {
  []
}

fn numeric_op_1(prim_fn: Prim1Fn, n) {
  prim_fn.numeric_fn
  |> lower_float1
  |> extend_rank1_numeric(n, scalar_shape_fn_1)
}

fn numeric_op_2(prim_fn: Prim2Fn, m, n) {
  prim_fn.numeric_fn
  |> lower_float2
  |> extend_rank2_numeric(m, n, scalar_shape_fn_2)
}

pub fn multiply_numeric(t, u) {
  { multiply_0_0() |> numeric_op_2(0, 0) }(t, u)
}

pub fn subtract_numeric(t, u) {
  { subtract_0_0() |> numeric_op_2(0, 0) }(t, u)
}

pub fn divide_numeric(t, u) {
  { divide_0_0() |> numeric_op_2(0, 0) }(t, u)
}

pub fn expt_numeric(t, u) {
  { expt_0_0() |> numeric_op_2(0, 0) }(t, u)
}

pub fn exp_numeric(t) {
  { exp_0() |> numeric_op_1(0) }(t)
}

pub fn log_numeric(t) {
  { log_0() |> numeric_op_1(0) }(t)
}

pub fn abs_numeric(t) {
  { abs_0() |> numeric_op_1(0) }(t)
}

pub fn rectify_numeric(t) {
  { rectify_0() |> numeric_op_1(0) }(t)
}

pub fn sqrt_numeric(t) {
  expt_numeric(t, float_to_tensor(0.5))
}

pub fn sqr_numeric(t) {
  multiply_numeric(t, t)
}

//----------------------------
// Boolean comparators
//----------------------------

// pub fn as_float(t: Tensor) -> Float {
//   let omit_bits = t.offset * 64
//   let assert <<_omit:size(omit_bits), value:float, _:bits>> = t.store
//   value
// }

// pub fn lift_float_to_comparator(f) {
//   fn(ta: Tensor, tb: Tensor) { f(ta |> as_float, tb |> as_float) }
// }

// pub fn comparator(f) {
//   fn(ta: Tensor, tb: Tensor) -> Bool {
//     { f |> lift_float_to_comparator }(ta, tb)
//   }
// }

// pub fn equal_rank_0_0() {
//   fn(x, y) { x == y } |> comparator
// }

// pub fn less_rank_0_0() {
//   fn(x, y) { x <. y } |> comparator
// }

// pub fn less_equal_rank_0_0() {
//   fn(x, y) { x <=. y } |> comparator
// }

// pub fn great_rank_0_0() {
//   fn(x, y) { x >. y } |> comparator
// }

// pub fn great_equal_rank_0_0() {
//   fn(x, y) { x >=. y } |> comparator
// }

//----------------------------
// Tensorized comparators
//----------------------------

pub fn lower_float2_comparison(
  f: fn(Float, Float) -> Bool,
) -> fn(BitArray, BitArray) -> BitArray {
  fn(a_slice, b_slice) {
    let assert <<a:float>> = a_slice
    let assert <<b:float>> = b_slice
    let c = f(a, b)
    case c {
      True -> <<1.0:float>>
      False -> <<0.0:float>>
    }
  }
}

pub fn t_great_than(t, u) {
  let f =
    fn(x, y) { x >. y }
    |> lower_float2_comparison
    |> extend_rank2_numeric(0, 0, scalar_shape_fn_2)

  f(t, u)
}

pub fn t_less_than(t, u) {
  let f =
    fn(x, y) { x <. y }
    |> lower_float2_comparison
    |> extend_rank2_numeric(0, 0, scalar_shape_fn_2)

  f(t, u)
}

pub fn t_equal(t, u) {
  let f =
    fn(x, y) { x == y }
    |> lower_float2_comparison
    |> extend_rank2_numeric(0, 0, scalar_shape_fn_2)

  f(t, u)
}
