import flat_tensor.{
  type Differentiable, type Dual, type Shape, type Tensor, Dual, DualDiff,
  ListDiff, add_numeric, bitarray_to_floats, build_store, build_tensor, d_add,
  d_divide, d_exp, d_expt, d_log, d_multiply, d_sqr, d_sqrt, d_subtract,
  equal_elements, extend_rank1_gradient, extend_rank1_numeric,
  extend_rank2_gradient, extend_rank2_numeric, extend_rank2_shapes,
  float_bits_walker, float_to_tensor, floats_to_tensor, gradient_of,
  gradient_once, idxs, lower_float2, lower_float3, map_tensor_recursively,
  merge_shapes, min_shape, new_flat, rank, reshape, shape, size_of, store,
  tensor_equal, tlen, to_bitarray, to_diff, to_dual, to_tensor, tref, trefs,
  unwrap_ok_number, unwrap_ok_number2,
}
import gleam/bit_array
import gleam/bool
import gleam/dynamic.{type Dynamic}
import gleam/float
import gleam/int
import gleam/list
import gleam_community/maths/elementary.{exponential, natural_logarithm}
import gleeunit/should

pub fn scalar1_shape(_) {
  []
}

pub fn scalar2_shape(_, _) {
  []
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
  let r1_td = [3, 4, 5] |> dynamic.from |> to_tensor
  let r3_td =
    [
      [[0, 1], [2, 3], [4, 5]],
      [[6, 7], [8, 9], [10, 11]],
      [[12, 13], [14, 15], [16, 17]],
      [[18, 19], [20, 21], [22, 23]],
    ]
    |> dynamic.from
    |> to_tensor

  r1_td |> tref(2) |> tensor_should_equal(5.0 |> float_to_tensor)
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
  |> tensor_should_equal([[[0, 1, 2], [1, 2, 3]]] |> dynamic.from |> to_tensor)

  r1_td
  |> trefs([0, 2])
  |> tensor_should_equal([3, 5] |> dynamic.from |> to_tensor)
}

pub fn tensor_operations_test() {
  let r0_td = float_to_tensor(3.0)
  let r1_td = [3, 4, 5] |> dynamic.from |> to_tensor
  let r2_td =
    [[3, 4, 5], [7, 8, 9]]
    |> dynamic.from
    |> to_tensor
  let r3_td =
    [
      [[0, 1], [2, 3], [4, 5]],
      [[6, 7], [8, 9], [10, 11]],
      [[12, 13], [14, 15], [16, 17]],
      [[18, 19], [20, 21], [22, 23]],
    ]
    |> dynamic.from
    |> to_tensor

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
    |> to_tensor,
  )
}

pub fn extend_ops_extend_rank1_numeric_test() {
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

    t |> extend_rank1_numeric(sum_f, 1, sum_shape_f)
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
        [x, ..] -> [x * 2]
        _ -> panic as "Invalid shape for dup"
      }
    }

    t |> extend_rank1_numeric(dup_f, 1, dup_shape_f)
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

pub fn extend_ops_extend_rank2_shapes_test() {
  let s0 = [3, 4, 5, 6]
  let s1 = [3, 7, 6]
  let r0 = 2
  let r1 = 1

  extend_rank2_shapes(
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

pub fn extend_ops_extend_rank2_numeric_test() {
  let multiply_numeric =
    float.multiply |> lower_float2 |> extend_rank2_numeric(0, 0, scalar2_shape)

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

pub fn extend_ops_extend_rank2_multiply_2_1_test() {
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
  // The mul_2_1 function then uses this mul_2_1_f with extend_rank2_numeric,
  // specifying the minimum ranks (2 and 1) and the output shape function.
  let mul_2_1 = mul_2_1_f |> extend_rank2_numeric(2, 1, fn(s0, _s1) { s0 })

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
  let r1_td = new_flat([3], [3.0, 4.0, 5.0] |> to_bitarray, 0)
  let r2_td = new_flat([2, 3], [3.0, 4.0, 5.0, 7.0, 8.0, 9.0] |> to_bitarray, 0)

  let one_like = fn(t: Tensor) {
    new_flat(t.shape, list.repeat(1.0, size_of(t.shape)) |> to_bitarray, 0)
  }

  {
    let sqr_gradient = fn(a, z) { z *. 2.0 *. a }

    let tensor_sqr =
      sqr_gradient |> lower_float2 |> extend_rank1_gradient(0, scalar1_shape)

    tensor_sqr(r1_td, r1_td |> one_like)
    |> store
    |> should.equal([6.0, 8.0, 10.0] |> to_bitarray)

    let gsqr = tensor_sqr(r2_td, r2_td |> one_like)
    gsqr.shape |> should.equal([2, 3])
    gsqr.store
    |> should.equal([6.0, 8.0, 10.0, 14.0, 16.0, 18.0] |> to_bitarray)
  }

  let add_gradient = fn(_a, _b, z) { #(z, z) }
  let tensor_add = add_gradient |> extend_rank2_gradient(0, 0, scalar2_shape)
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
      |> extend_rank2_gradient(0, 0, scalar2_shape)

    let #(gt, gu) =
      multiply_gradient(
        [2, 3, 4] |> dynamic.from |> to_tensor,
        [1, 2, 3] |> dynamic.from |> to_tensor,
        [1, 1, 1] |> dynamic.from |> to_tensor,
      )

    gt |> tensor_should_equal([1, 2, 3] |> dynamic.from |> to_tensor)
    gu |> tensor_should_equal([2, 3, 4] |> dynamic.from |> to_tensor)
  }
  {
    let sum_1_gradient = fn(g: BitArray, vz: BitArray) -> BitArray {
      let assert <<z:float>> = vz
      float_bits_walker(fn(acc, _v) { <<acc:bits, z:float>> }, g, <<>>)
    }
    let sum_gradient = sum_1_gradient |> extend_rank1_gradient(1, scalar1_shape)

    sum_gradient([2, 3, 4] |> dynamic.from |> to_tensor, float_to_tensor(1.0))
    |> tensor_should_equal([1, 1, 1] |> dynamic.from |> to_tensor)

    sum_gradient(
      [[2, 3, 4], [2, 3, 4]]
        |> dynamic.from
        |> to_tensor,
      [2, 1] |> dynamic.from |> to_tensor,
    )
    |> tensor_should_equal(
      [[2, 2, 2], [1, 1, 1]]
      |> dynamic.from
      |> to_tensor,
    )
  }
}

fn differentiable_compare(x: Differentiable, y: Differentiable) {
  case x, y {
    DualDiff(a), DualDiff(b) -> tensor_equal(a.tensor, b.tensor)
    ListDiff(a), ListDiff(b) ->
      list.map2(a, b, differentiable_compare) |> list.fold(True, bool.and)
    _, _ -> False
  }
}

fn differentiable_to_floats(x: Differentiable) -> Dynamic {
  case x {
    DualDiff(a) -> a.tensor.store |> bitarray_to_floats |> dynamic.from
    ListDiff(a) -> a |> list.map(differentiable_to_floats) |> dynamic.from
  }
}

pub fn differentiable_should_equal(x: Differentiable, y: Differentiable) {
  case differentiable_compare(x, y) {
    True -> should.be_true(True)
    _ -> should.equal(differentiable_to_floats(x), differentiable_to_floats(y))
  }
}

fn as_dual(y: Differentiable) {
  let assert DualDiff(d) = y
  d
}

fn get_tensor(d: Dual) {
  d.tensor
}

pub fn map_tensor_recursively_test() {
  let t =
    [0.0, 1.0, 2.0, 3.0] |> dynamic.from |> to_tensor |> to_dual |> DualDiff

  fn(d: Dual) {
    Dual(d.id, d.tensor |> add_numeric(float_to_tensor(1.0)), d.link)
  }
  |> map_tensor_recursively(t)
  |> as_dual
  |> get_tensor
  |> tensor_should_equal([1.0, 2.0, 3.0, 4.0] |> dynamic.from |> to_tensor)
}

pub fn autodiff_test() {
  let dual0 = float_to_tensor(0.0) |> to_dual
  let dual1 = float_to_tensor(1.0) |> to_dual

  map_tensor_recursively(
    fn(d: Dual) { d },
    gradient_once(
      dual1 |> DualDiff,
      [dual0, dual1] |> list.map(fn(x) { DualDiff(x) }) |> ListDiff,
    ),
  )
  |> differentiable_should_equal([0.0, 1.0] |> dynamic.from |> to_diff)
}

pub fn check_gradients1(f, t, gradients) {
  let f_wrapper = fn(lst) {
    let assert ListDiff([a]) = lst
    f(a |> as_dual) |> DualDiff
  }
  let theta = [t] |> list.map(fn(x) { x |> to_dual |> DualDiff }) |> ListDiff

  gradient_of(f_wrapper, theta) |> differentiable_should_equal(gradients)
}

pub fn check_gradients2(f, t, u, gradients) {
  let f_wrapper = fn(lst) {
    let assert ListDiff([a, b]) = lst
    f(a |> as_dual, b |> as_dual) |> DualDiff
  }
  let theta = [t, u] |> list.map(fn(x) { x |> to_dual |> DualDiff }) |> ListDiff

  gradient_of(f_wrapper, theta) |> differentiable_should_equal(gradients)
}

pub fn check_theta1(f, t, answers) {
  f(t |> to_dual) |> get_tensor |> tensor_should_equal(answers)
}

pub fn check_theta2(f, t, u, answers) {
  f(t |> to_dual, u |> to_dual) |> get_tensor |> tensor_should_equal(answers)
}

pub fn check_theta_and_gradient1(f) {
  fn(t: Tensor, answers: Tensor, gradients: Differentiable) {
    check_theta1(f, t, answers)
    check_gradients1(f, t, gradients)
  }
}

pub fn check_theta_and_gradient2(f) {
  fn(t: Tensor, u: Tensor, answers: Tensor, gradients: Differentiable) {
    check_theta2(f, t, u, answers)
    check_gradients2(f, t, u, gradients)
  }
}

pub fn a_scalar_ops_test() {
  let a = float_to_tensor(2.0)
  let b = float_to_tensor(3.0)

  { d_add |> check_theta_and_gradient2 }(
    a,
    b,
    float_to_tensor(5.0),
    [1, 1] |> dynamic.from |> to_diff,
  )
  { d_subtract |> check_theta_and_gradient2 }(
    a,
    b,
    float_to_tensor(-1.0),
    [1, -1] |> dynamic.from |> to_diff,
  )
  { d_multiply |> check_theta_and_gradient2 }(
    a,
    b,
    float_to_tensor(6.0),
    [3.0, 2.0] |> dynamic.from |> to_diff,
  )
  { d_divide |> check_theta_and_gradient2 }(
    a,
    b,
    float_to_tensor(2.0 /. 3.0),
    [0.3333333333333333, -0.2222222222222222]
      |> dynamic.from
      |> to_diff,
  )
  { d_exp |> check_theta_and_gradient1 }(
    a,
    exponential(2.0) |> float_to_tensor,
    [exponential(2.0)] |> dynamic.from |> to_diff,
  )
  { d_log |> check_theta_and_gradient1 }(
    a,
    unwrap_ok_number(natural_logarithm, 2.0) |> float_to_tensor,
    [0.5] |> dynamic.from |> to_diff,
  )
  { d_expt |> check_theta_and_gradient2 }(
    a,
    b,
    float_to_tensor(8.0),
    [12.0, 5.545177444479562] |> dynamic.from |> to_diff,
  )
  { d_sqrt |> check_theta_and_gradient1 }(
    a,
    unwrap_ok_number(float.square_root, 2.0) |> float_to_tensor,
    [0.3535533905932738] |> dynamic.from |> to_diff,
  )
  check_gradients1(
    d_sqr,
    float_to_tensor(3.0),
    [6.0] |> dynamic.from |> to_diff,
  )

  check_gradients2(
    fn(x, y) { d_add(d_log(x), d_multiply(x, y)) },
    float_to_tensor(2.0),
    float_to_tensor(2.0),
    [2.5, 2.0] |> dynamic.from |> to_diff,
  )
}

pub fn a_scalar_ops_numeric_test() {
  // Check numericals with vector-duals
  let a = [2, 3, 4] |> dynamic.from |> to_tensor
  let b = [3, 8, 9] |> dynamic.from |> to_tensor
  { d_add |> check_theta_and_gradient2 }(
    a,
    b,
    [5, 11, 13] |> dynamic.from |> to_tensor,
    [[1, 1, 1], [1, 1, 1]]
      |> dynamic.from
      |> to_diff,
  )
  { d_subtract |> check_theta_and_gradient2 }(
    a,
    b,
    [-1, -5, -5] |> dynamic.from |> to_tensor,
    [[1, 1, 1], [-1, -1, -1]]
      |> dynamic.from
      |> to_diff,
  )
  { d_multiply |> check_theta_and_gradient2 }(
    a,
    b,
    [6, 24, 36]
      |> dynamic.from
      |> to_tensor,
    [[3, 8, 9], [2, 3, 4]]
      |> dynamic.from
      |> to_diff,
  )
  { d_divide |> check_theta_and_gradient2 }(
    a,
    b,
    [2.0 /. 3.0, 3.0 /. 8.0, 4.0 /. 9.0]
      |> dynamic.from
      |> to_tensor,
    [
      [0.3333333333333333, 0.125, 0.1111111111111111],
      [-0.2222222222222222, -0.046875, -0.04938271604938271],
    ]
      |> dynamic.from
      |> to_diff,
  )
  { d_exp |> check_theta_and_gradient1 }(
    a,
    [exponential(2.0), exponential(3.0), exponential(4.0)]
      |> dynamic.from
      |> to_tensor,
    [[exponential(2.0), exponential(3.0), exponential(4.0)]]
      |> dynamic.from
      |> to_diff,
  )
  { d_log |> check_theta_and_gradient1 }(
    a,
    [2.0, 3.0, 4.0]
      |> list.map(unwrap_ok_number(natural_logarithm, _))
      |> dynamic.from
      |> to_tensor,
    [[0.5, 1.0 /. 3.0, 1.0 /. 4.0]]
      |> dynamic.from
      |> to_diff,
  )
  { d_expt |> check_theta_and_gradient2 }(
    a,
    b,
    [
      unwrap_ok_number2(float.power, 2.0, 3.0),
      unwrap_ok_number2(float.power, 3.0, 8.0),
      unwrap_ok_number2(float.power, 4.0, 9.0),
    ]
      |> dynamic.from
      |> to_tensor,
    [
      [12.0, 17_496.0, 589_824.0],
      [5.545177444479562, 7207.9952259514685, 363_408.7490014126],
    ]
      |> dynamic.from
      |> to_diff,
  )
  { d_sqrt |> check_theta_and_gradient1 }(
    a,
    [
      unwrap_ok_number(float.square_root, 2.0),
      unwrap_ok_number(float.square_root, 3.0),
      unwrap_ok_number(float.square_root, 4.0),
    ]
      |> dynamic.from
      |> to_tensor,
    [[0.3535533905932738, 0.28867513459481287, 0.25]]
      |> dynamic.from
      |> to_diff,
  )
  { d_sqr |> check_theta_and_gradient1 }(
    b,
    [9, 64, 81] |> dynamic.from |> to_tensor,
    [[6, 16, 18]]
      |> dynamic.from
      |> to_diff,
  )
  check_gradients2(
    fn(x, y) { d_add(d_log(x), d_multiply(x, y)) },
    [2, 3, 4] |> dynamic.from |> to_tensor,
    [2, 3, 4] |> dynamic.from |> to_tensor,
    [[2.5, 3.3333333333333335, 4.25], [2.0, 3.0, 4.0]]
      |> dynamic.from
      |> to_diff,
  )
  check_gradients2(
    fn(x, y) { d_add(d_log(x), d_multiply(x, y)) },
    a,
    a,
    [[2.5, 3.3333333333333335, 4.25], [2.0, 3.0, 4.0]]
      |> dynamic.from
      |> to_diff,
  )
}

pub fn a_scalar_ops_numeric_lists_test() {
  let duals_to_differentiable = fn(duals) {
    duals |> list.map(DualDiff(_)) |> ListDiff
  }
  let tensors_to_differentiable = fn(tensors) {
    tensors |> list.map(to_dual) |> duals_to_differentiable
  }
  let lift_op = fn(op: fn(Dual, Dual) -> Differentiable) {
    fn(theta) {
      let assert ListDiff([DualDiff(m), DualDiff(n)]) = theta
      op(m, n)
    }
  }
  let check_gradient_of = fn(op, wrt, gradients) {
    gradient_of(op |> lift_op, wrt |> tensors_to_differentiable)
    |> differentiable_should_equal(gradients)
  }

  // Check numericals with lists
  let x = float_to_tensor(3.0)
  let y = float_to_tensor(2.0)
  let f = d_multiply
  { d_multiply |> check_theta_and_gradient2 }(
    x,
    y,
    float_to_tensor(6.0),
    [2.0, 3.0] |> dynamic.from |> to_diff,
  )

  check_gradient_of(
    fn(m, n) { [f(m, m), f(n, n)] |> duals_to_differentiable },
    //
    [x, y],
    [6, 4] |> dynamic.from |> to_diff,
  )

  check_gradient_of(
    fn(m, n) {
      [
        [f(m, m), f(n, n)] |> duals_to_differentiable,
        [f(m, m), f(n, n)] |> duals_to_differentiable,
      ]
      |> ListDiff
    },
    //
    [x, y],
    [12, 8] |> dynamic.from |> to_diff,
  )

  check_gradient_of(
    fn(m, n) {
      [m, n] |> list.map(fn(x) { f(x, x) }) |> duals_to_differentiable
    },
    //
    [x, y],
    [6, 4] |> dynamic.from |> to_diff,
  )

  check_gradient_of(
    fn(m, n) { list.map2([m, n], [m, n], f) |> duals_to_differentiable },
    [x, y],
    [6, 4] |> dynamic.from |> to_diff,
  )

  check_gradient_of(
    fn(m, n) { list.map2([m, n], [n, m], f) |> duals_to_differentiable },
    //
    [x, y],
    [4, 6] |> dynamic.from |> to_diff,
  )
  {
    let a = float_to_tensor(7.0)
    let b = [13] |> dynamic.from |> to_tensor

    { d_add |> check_theta_and_gradient2 }(
      a,
      b,
      [20] |> dynamic.from |> to_tensor,
      [
        float_to_tensor(1.0),
        //
        [1.0] |> dynamic.from |> to_tensor,
      ]
        |> tensors_to_differentiable,
    )
    { d_multiply |> check_theta_and_gradient2 }(
      a,
      b,
      [91] |> dynamic.from |> to_tensor,
      [
        float_to_tensor(13.0),
        //
        [7.0] |> dynamic.from |> to_tensor,
      ]
        |> tensors_to_differentiable,
    )
    { d_divide |> check_theta_and_gradient2 }(
      a,
      b,
      [7.0 /. 13.0] |> dynamic.from |> to_tensor,
      [
        float_to_tensor(0.07692),
        //
        [-0.04142] |> dynamic.from |> to_tensor,
      ]
        |> tensors_to_differentiable,
    )
  }
  let a = float_to_tensor(7.0)
  let b = [13, 15] |> dynamic.from |> to_tensor

  { d_add |> check_theta_and_gradient2 }(
    a,
    b,
    [20, 22] |> dynamic.from |> to_tensor,
    [
      float_to_tensor(2.0),
      //
      [1, 1] |> dynamic.from |> to_tensor,
    ]
      |> tensors_to_differentiable,
  )
  { d_multiply |> check_theta_and_gradient2 }(
    a,
    b,
    [91, 105] |> dynamic.from |> to_tensor,
    [
      float_to_tensor(28.0),
      //
      [7, 7] |> dynamic.from |> to_tensor,
    ]
      |> tensors_to_differentiable,
  )
  { d_divide |> check_theta_and_gradient2 }(
    a,
    b,
    [7.0 /. 13.0, 7.0 /. 15.0] |> dynamic.from |> to_tensor,
    [
      float_to_tensor(0.14358),
      //
      [-0.04142, -0.03111] |> dynamic.from |> to_tensor,
    ]
      |> tensors_to_differentiable,
  )
}
