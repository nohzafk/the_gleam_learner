import gleam/bit_array
import gleam/dynamic
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleeunit/should
import malt1.{
  type Shape, type Tensor, build_store, build_tensor, equal_elements, ext1_rho,
  flat_ext1_rho, float_bits_walker, floats_to_tensor, merge_shapes, min_shape,
  new_flat, rank, reshape, shape, size_of, tensor, tensor_equal, tlen, to_tensor,
  tref, trefs,
}

fn tensor_should_equal(actual, expected) {
  case tensor_equal(actual, expected) {
    True -> should.be_true(True)
    False -> should.equal(actual, expected)
  }
}

pub fn equality_test() {
  let t0 =
    new_flat([2, 3, 4], build_store(24, fn(i) { 2.0 *. int.to_float(i) }), 0)

  let t1 =
    new_flat(
      [2, 3, 4],
      build_store(24, fn(i) { 2.000001 *. int.to_float(i) }),
      0,
    )

  let t2 =
    new_flat([1, 2, 3, 4], build_store(24, fn(i) { 2.0 *. int.to_float(i) }), 0)

  let t3 =
    new_flat(
      [2, 2, 3, 4],
      build_store(48, fn(i) { int.to_float({ i / 24 } * i) }),
      0,
    )

  let t4 =
    new_flat(
      [2, 2, 3, 4],
      build_store(48, fn(i) {
        2.000001 *. int.to_float({ { i / 24 } * i }) -. 48.0
      }),
      0,
    )

  equal_elements(t0, t1) |> should.be_true
  // elements are equal, but shapes are not
  equal_elements(t0, t2) |> should.be_true

  equal_elements(t0, new_flat([2, 3, 4], t2.store, 0))

  equal_elements(t1, new_flat([2, 3, 4], t3.store, 24))

  equal_elements(t1, new_flat([2, 3, 4], t4.store, 24))

  tensor_equal(t0, t1) |> should.be_true

  tensor_equal(t0, t2) |> should.be_false

  tensor_equal(t0, new_flat([2, 3, 4], t2.store, 0))

  tensor_equal(t1, new_flat([2, 3, 4], t3.store, 24))

  tensor_equal(t1, new_flat([2, 3, 4], t4.store, 24))
}

pub fn tensor_basics_test() {
  let r1_td = [3, 4, 5] |> dynamic.from |> tensor
  let r3_td =
    [
      [[0, 1], [2, 3], [4, 5]],
      [[6, 7], [8, 9], [10, 11]],
      [[12, 13], [14, 15], [16, 17]],
      [[18, 19], [20, 21], [22, 23]],
    ]
    |> dynamic.from
    |> tensor

  r1_td |> tref(2) |> tensor_should_equal(to_tensor(5.0))
  r1_td |> tlen |> should.equal(3)
  [3.0, 4.0, 5.0] |> floats_to_tensor |> tensor_should_equal(r1_td)

  build_tensor([4, 3, 2], fn(idx) {
    let assert [a, b, c] = idx
    { 6 * a + 2 * b + c } |> int.to_float
  })
  |> tensor_should_equal(r3_td)

  build_tensor([1, 2, 3], fn(idx) {
    let assert [a, b, c] = idx
    { a + b + c } |> int.to_float
  })
  |> tensor_should_equal([[[0, 1, 2], [1, 2, 3]]] |> dynamic.from |> tensor)

  r1_td
  |> trefs([0, 2])
  |> tensor_should_equal([3, 5] |> dynamic.from |> tensor)
}

pub fn tensor_operations_test() {
  let r0_td = to_tensor(3.0)
  let r1_td = [3, 4, 5] |> dynamic.from |> tensor
  let r2_td =
    [[3, 4, 5], [7, 8, 9]]
    |> dynamic.from
    |> tensor
  let r3_td =
    [
      [[0, 1], [2, 3], [4, 5]],
      [[6, 7], [8, 9], [10, 11]],
      [[12, 13], [14, 15], [16, 17]],
      [[18, 19], [20, 21], [22, 23]],
    ]
    |> dynamic.from
    |> tensor

  r0_td |> shape |> should.equal([])
  r1_td |> shape |> should.equal([3])
  r2_td |> shape |> should.equal([2, 3])

  r0_td |> rank |> should.equal(0)
  r1_td |> rank |> should.equal(1)
  r2_td |> rank |> should.equal(2)

  size_of([]) |> should.equal(1)
  size_of([2, 2, 3]) |> should.equal(12)
  size_of([4, 3, 2]) |> should.equal(24)

  reshape(r3_td, [24])
  |> tensor_should_equal(
    [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23,
    ]
    |> dynamic.from
    |> tensor,
  )
}

fn bit_array_to_floats(slice: BitArray) {
  float_bits_walker(fn(acc, i) { [i, ..acc] }, slice, []) |> list.reverse
}

pub fn extend_ops_test() {
  min_shape(2, [3, 4, 5, 6]) |> should.equal([5, 6])
  min_shape(0, [3, 4, 5, 6]) |> should.equal([])

  merge_shapes([3, 4, 5, 6], 1, []) |> should.equal([3, 4, 5])

  let t0 =
    new_flat([2, 3, 4], build_store(24, fn(i) { { 2 * i } |> int.to_float }), 0)

  let sum = fn(t: Tensor) {
    let sum_f = fn(slice: BitArray) -> BitArray {
      let sum = float_bits_walker(fn(acc, i) { acc +. i }, slice, 0.0)
      <<sum:float>>
    }

    let sum_shape_f = fn(_in_f_shape: Shape) -> Shape { [] }

    t |> ext1_rho(sum_f, 1, sum_shape_f)
  }
  let sum_t = sum(t0)
  sum_t.store
  |> bit_array_to_floats
  |> should.equal([12.0, 44.0, 76.0, 108.0, 140.0, 172.0])

  let dup = fn(t: Tensor) {
    let dup_f = fn(slice: BitArray) -> BitArray {
      bit_array.concat([slice, slice])
    }

    let dup_shape_f = fn(in_f_shape: Shape) -> Shape {
      case in_f_shape {
        [x, ..rest] -> [x * 2, ..rest]
        _ -> panic as "Invalid shape for dup"
      }
    }

    t |> ext1_rho(dup_f, 1, dup_shape_f)
  }

  let dup_t = dup(t0)
  dup_t.store
  |> bit_array_to_floats
  |> should.equal(
    [
      0, 2, 4, 6, 0, 2, 4, 6, 8, 10, 12, 14, 8, 10, 12, 14, 16, 18, 20, 22, 16,
      18, 20, 22, 24, 26, 28, 30, 24, 26, 28, 30, 32, 34, 36, 38, 32, 34, 36, 38,
      40, 42, 44, 46, 40, 42, 44, 46,
    ]
    |> list.map(int.to_float),
  )
}
