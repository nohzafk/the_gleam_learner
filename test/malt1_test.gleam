import gleam/bit_array
import gleam/dynamic
import gleam/erlang/atom
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleeunit/should
import malt0
import malt1.{
  type Shape, type Tensor, bitarray_to_floats, build_store, build_tensor,
  equal_elements, ext1_gradient, ext1_numeric, ext2_gradient, ext2_numeric,
  ext2_shapes, float_bits_walker, floats_to_tensor, idxs, lower_float2,
  lower_float3, merge_shapes, min_shape, new_flat, rank, reshape, shape, size_of,
  store, tensor, tensor_equal, tlen, to_bitarray, to_tensor, tref, trefs,
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

pub fn extend_ops_ext1_numeric_test() {
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

    t |> ext1_numeric(sum_f, 1, sum_shape_f)
  }
  let sum_t = sum(t0)
  sum_t.store
  |> bitarray_to_floats
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

    t |> ext1_numeric(dup_f, 1, dup_shape_f)
  }

  let dup_t = dup(t0)
  dup_t.store
  |> bitarray_to_floats
  |> should.equal(
    [
      0, 2, 4, 6, 0, 2, 4, 6, 8, 10, 12, 14, 8, 10, 12, 14, 16, 18, 20, 22, 16,
      18, 20, 22, 24, 26, 28, 30, 24, 26, 28, 30, 32, 34, 36, 38, 32, 34, 36, 38,
      40, 42, 44, 46, 40, 42, 44, 46,
    ]
    |> list.map(int.to_float),
  )
}

pub fn extend_ops_ext2_shapes_test() {
  let s0 = [3, 4, 5, 6]
  let s1 = [3, 7, 6]
  let r0 = 2
  let r1 = 1

  ext2_shapes(
    s0,
    s1,
    r0,
    r1,
    [5, 6],
    fn(s_out: Shape, size_out: Int, _q0: Int, _q1: Int, strides) {
      s_out |> should.equal([3, 4, 7, 5, 6])
      size_out |> should.equal(2520)
      strides
      |> should.equal([#(840, 120, 42), #(210, 30, 0), #(30, 0, 6)])

      {
        let #(i0, i1) = idxs(strides, 30, 0, 0)
        i0 |> should.equal(0)
        i1 |> should.equal(6)
      }
      {
        let #(i0, i1) = idxs(strides, 210, 0, 0)
        i0 |> should.equal(30)
        i1 |> should.equal(0)
      }
      {
        let #(i0, i1) = idxs(strides, 240, 0, 0)
        i0 |> should.equal(30)
        i1 |> should.equal(6)
      }
      {
        let #(i0, i1) = idxs(strides, 420, 0, 0)
        i0 |> should.equal(60)
        i1 |> should.equal(0)
      }
      {
        let #(i0, i1) = idxs(strides, 840, 0, 0)
        i0 |> should.equal(120)
        i1 |> should.equal(42)
      }
    },
  )
}

pub fn extend_ops_ext2_numeric_test() {
  let multiply_numeric =
    float.multiply |> lower_float2 |> ext2_numeric(0, 0, malt1.scalar2_shape)

  let t0 =
    new_flat([2, 3, 4], build_store(24, fn(i) { { 2 * i } |> int.to_float }), 0)
  let t0sqr = multiply_numeric(t0, t0)
  t0sqr.store
  |> bitarray_to_floats
  |> should.equal(
    [
      0, 4, 16, 36, 64, 100, 144, 196, 256, 324, 400, 484, 576, 676, 784, 900,
      1024, 1156, 1296, 1444, 1600, 1764, 1936, 2116,
    ]
    |> list.map(int.to_float),
  )
}

pub fn extend_ops_ext2_multiply_2_1_test() {
  let t1 =
    new_flat([5, 6], build_store(30, fn(i) { 2.0 *. int.to_float(i) }), 0)
  let t2 = new_flat([6], build_store(6, fn(i) { 3.0 *. int.to_float(i) }), 0)

  // The *-2-1-f function is performing an element-wise multiplication with broadcasting.
  //
  // 1. (modulo j0 s1) is used for broadcasting. It allows the second tensor (v1) to be repeated along its first dimension if it's smaller than the first tensor (v0).
  // 2. The function multiplies elements from v0 and v1, potentially repeating v1 if it's smaller.
  //
  // (define *-2-1-f
  //  (λ (v0 i0 s0 v1 i1 s1 vout iout sout)
  //    (for ([j0 (in-range 0 s0)])
  //      (vset! vout (+ iout j0)
  //        (* (vref v0 (+ i0 j0))
  //           (vref v1 (+ i1 (modulo j0 s1))))))))
  //
  //
  // 1. We calculate the sizes s0 and s1 based on the byte size of the input bit arrays.
  // 2. We use list.range to iterate over the elements of the first tensor.
  // 3. For each element, we extract the float from t0_slice and the corresponding (potentially broadcasted) float from t1_slice.
  // 4. We multiply these floats and create a new float bit array.
  // 5. Finally, we concatenate all these individual float bit arrays.
  let mul_2_1_f = fn(t0_slice: BitArray, t1_slice: BitArray) -> BitArray {
    // how many float (a float is 64 bits, which is 8 bytes)
    let s0 = bit_array.byte_size(t0_slice) / 8
    let s1 = bit_array.byte_size(t1_slice) / 8

    list.range(0, s0 - 1)
    |> list.map(fn(j0) {
      let assert Ok(<<v0:float>>) = bit_array.slice(t0_slice, j0 * 8, 8)
      let assert Ok(<<v1:float>>) = bit_array.slice(t1_slice, j0 % s1 * 8, 8)
      let r = v0 *. v1
      <<r:float>>
    })
    |> bit_array.concat
  }
  // The mul_2_1 function then uses this mul_2_1_f with ext2_numeric,
  // specifying the minimum ranks (2 and 1) and the output shape function.
  let mul_2_1 = mul_2_1_f |> ext2_numeric(2, 1, fn(s0, _s1) { s0 })

  let r_1_2 = mul_2_1(t1, t2)
  r_1_2.shape |> should.equal([5, 6])
  r_1_2.store
  |> bitarray_to_floats
  |> should.equal([
    0.0, 6.0, 24.0, 54.0, 96.0, 150.0, 0.0, 42.0, 96.0, 162.0, 240.0, 330.0, 0.0,
    78.0, 168.0, 270.0, 384.0, 510.0, 0.0, 114.0, 240.0, 378.0, 528.0, 690.0,
    0.0, 150.0, 312.0, 486.0, 672.0, 870.0,
  ])

  let t3 =
    new_flat([3, 5, 6], build_store(90, fn(i) { 2.0 *. int.to_float(i) }), 0)
  let t4 =
    new_flat([3, 6], build_store(18, fn(i) { 3.0 *. int.to_float(i) }), 0)

  let r_3_4 = mul_2_1(t3, t4)
  r_3_4.shape |> should.equal([3, 5, 6])
  r_3_4.store
  |> bitarray_to_floats
  |> should.equal([
    0.0, 6.0, 24.0, 54.0, 96.0, 150.0, 0.0, 42.0, 96.0, 162.0, 240.0, 330.0, 0.0,
    78.0, 168.0, 270.0, 384.0, 510.0, 0.0, 114.0, 240.0, 378.0, 528.0, 690.0,
    0.0, 150.0, 312.0, 486.0, 672.0, 870.0,
    //
    1080.0, 1302.0, 1536.0, 1782.0, 2040.0, 2310.0, 1296.0, 1554.0, 1824.0,
    2106.0, 2400.0, 2706.0, 1512.0, 1806.0, 2112.0, 2430.0, 2760.0, 3102.0,
    1728.0, 2058.0, 2400.0, 2754.0, 3120.0, 3498.0, 1944.0, 2310.0, 2688.0,
    3078.0, 3480.0, 3894.0,
    //
    4320.0, 4758.0, 5208.0, 5670.0, 6144.0, 6630.0, 4752.0, 5226.0, 5712.0,
    6210.0, 6720.0, 7242.0, 5184.0, 5694.0, 6216.0, 6750.0, 7296.0, 7854.0,
    5616.0, 6162.0, 6720.0, 7290.0, 7872.0, 8466.0, 6048.0, 6630.0, 7224.0,
    7830.0, 8448.0, 9078.0,
  ])
}

pub fn extend_ops_ext_gradient_test() {
  let r0_td = to_tensor(3.0)
  let r1_td = new_flat([3], [3.0, 4.0, 5.0] |> to_bitarray, 0)
  let r2_td = new_flat([2, 3], [3.0, 4.0, 5.0, 7.0, 8.0, 9.0] |> to_bitarray, 0)

  let one_like = fn(t: Tensor) {
    new_flat(t.shape, list.repeat(1.0, size_of(t.shape)) |> to_bitarray, 0)
  }

  {
    let sqr_numeric = fn(a) { a *. a }
    let sqr_gradient = fn(a, z) { z *. 2.0 *. a }

    let tensor_sqr =
      sqr_gradient |> lower_float2 |> ext1_gradient(0, malt1.scalar1_shape)

    tensor_sqr(r1_td, r1_td |> one_like)
    |> store
    |> should.equal([6.0, 8.0, 10.0] |> to_bitarray)

    let gsqr = tensor_sqr(r2_td, r2_td |> one_like)
    gsqr.shape |> should.equal([2, 3])
    gsqr.store
    |> should.equal([6.0, 8.0, 10.0, 14.0, 16.0, 18.0] |> to_bitarray)
  }

  let add_numeric = float.add
  let add_gradient = fn(_a, _b, z) { #(z, z) }
  let tensor_add = add_gradient |> ext2_gradient(0, 0, malt1.scalar2_shape)
  {
    let #(ta, tb) = tensor_add(r1_td, r1_td, r1_td |> one_like)
    ta.shape |> should.equal([3])
    ta.store |> bitarray_to_floats |> should.equal([1.0, 1.0, 1.0])
    tb.shape |> should.equal([3])
    tb.store |> bitarray_to_floats |> should.equal([1.0, 1.0, 1.0])
  }
  {
    let #(ta, tb) = tensor_add(r1_td, r2_td, r2_td |> one_like)
    ta.shape |> should.equal([3])
    // ta.store |> should.equal([2.0, 2.0, 2.0] |> to_bitarray)
    ta.store |> bitarray_to_floats |> should.equal([2.0, 2.0, 2.0])
    tb.shape |> should.equal([2, 3])
    tb.store
    |> bitarray_to_floats
    |> should.equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
  }
  {
    let multiply_gradient =
      fn(a, b, z) { #(z *. b, z *. a) }
      |> lower_float3
      |> ext2_gradient(0, 0, malt1.scalar2_shape)

    let #(gt, gu) =
      multiply_gradient(
        [2, 3, 4] |> dynamic.from |> tensor,
        [1, 2, 3] |> dynamic.from |> tensor,
        [1, 1, 1] |> dynamic.from |> tensor,
      )

    gt |> tensor_should_equal([1, 2, 3] |> dynamic.from |> tensor)
    gu |> tensor_should_equal([2, 3, 4] |> dynamic.from |> tensor)
  }
  {
    let sum_1_gradient = fn(g: BitArray, vz: BitArray) -> BitArray {
      let assert <<z:float>> = vz
      float_bits_walker(fn(acc, v) { <<acc:bits, z:float>> }, g, <<>>)
    }
    let sum_gradient = sum_1_gradient |> ext1_gradient(1, malt1.scalar1_shape)

    sum_gradient([2, 3, 4] |> dynamic.from |> tensor, to_tensor(1.0))
    |> tensor_should_equal([1, 1, 1] |> dynamic.from |> tensor)

    sum_gradient(
      [[2, 3, 4], [2, 3, 4]]
        |> dynamic.from
        |> tensor,
      [2, 1] |> dynamic.from |> tensor,
    )
    |> tensor_should_equal(
      [[2, 2, 2], [1, 1, 1]]
      |> dynamic.from
      |> tensor,
    )
  }
}
