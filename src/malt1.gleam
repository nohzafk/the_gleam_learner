//// Flat binary tensor implementation

import gleam/bit_array
import gleam/dynamic.{type Dynamic}
import gleam/erlang.{type Reference}
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/result
import gleam/string
import mat

//----------------------------
//  Flat binary tensor
//----------------------------
pub const tolerace = 0.0001

pub type Shape =
  List(Int)

pub fn tensor_id() {
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
    id: tensor_id(),
    shape: shape,
    store: store,
    offset: offset,
    size: size_of(shape),
    strides: strides(shape),
    rank: list.length(shape),
  )
}

pub fn to_tensor(v: Float) {
  FlatTensor(
    id: tensor_id(),
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
      to_tensor(v)
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

pub fn tensor(lst: Dynamic) -> Tensor {
  case dynamic.classify(lst) {
    "Int" -> {
      use x <- result.map(lst |> dynamic.int)
      x |> int.to_float |> to_tensor
    }
    "Float" -> {
      use x <- result.map(lst |> dynamic.float)
      x |> to_tensor
    }
    "List" -> {
      use elements <- result.try(lst |> dynamic.list(dynamic.dynamic))
      let tensors = elements |> list.map(tensor)
      tensors |> merge_flat_tensors |> Ok
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
// Unary Pointwise extension
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

pub fn ext1_rho(f, m, shape_fn) {
  fn(t: Tensor) {
    case t.rank {
      0 -> f(t.store) |> scalarize |> to_tensor
      _ -> flat_ext1_rho(f, m, shape_fn, t)
    }
  }
}

pub fn flat_ext1_rho(
  f: fn(BitArray) -> BitArray,
  min_rank: Int,
  shape_fn: fn(Shape) -> Shape,
  t0: Tensor,
) -> Tensor {
  let s0 = t0.shape
  let sf0 = min_shape(min_rank, s0)
  let stride0 = size_of(sf0)
  let s_out = merge_shapes(s0, min_rank, shape_fn(sf0))

  let v_out =
    list.range(0, t0.size / stride0 - 1)
    |> list.map(fn(i0) {
      // slice use bytes as unit
      let assert Ok(slice) =
        t0.store |> bit_array.slice(t0.offset + i0 * stride0 * 8, stride0 * 8)
      f(slice)
    })
    |> bit_array.concat

  new_flat(shape: s_out, store: v_out, offset: 0)
}
