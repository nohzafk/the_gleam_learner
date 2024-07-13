import gleam/dict
import gleam/dynamic
import gleam/float
import gleam/int
import gleam/result
import gleeunit/should

import learner.{
  type Scalar, ListTensor, Scalar, ScalarTensor, build_tensor,
  build_tensor_from_tensors, correlation_overlap, desc_t, desc_u, dot_product,
  dotted_product, ext1, ext2, get_real, gradient_of, gradient_once,
  is_tensor_equal, map_to_scalar, of_rank, of_ranks, rank, rectify, rectify_0,
  shape, size_of, sum_dp, tensor, tensor1_map, tensor_argmax, tensor_correlate,
  tensor_max, tensor_minus, tensor_multiply, tensor_multiply_2_1, tensor_sqr,
  tensor_sum, tensor_sum_cols, tensor_to_list, to_scalar,
}
import test_utils.{check_theta_and_gradient1, check_theta_and_gradient2}

pub fn dual_as_key_test() {
  let d1 = to_scalar(0.1)
  let v1 = 1.0
  let sigma = dict.new() |> dict.insert(d1, v1)

  sigma |> dict.size() |> should.equal(1)
  sigma |> dict.get(d1) |> result.unwrap(0.0) |> should.equal(v1)

  let d2 = to_scalar(0.1)
  let v2 = 2.0
  let sigma2 = sigma |> dict.insert(d2, v2)

  sigma2 |> dict.size() |> should.equal(2)
  sigma2 |> dict.get(d2) |> result.unwrap(0.0) |> should.equal(v2)
}

pub fn build_tensor_test() {
  let t1 =
    build_tensor([1, 2, 3], fn(idx) {
      let assert [a, b, c] = idx
      { a + b + c } |> int.to_float
    })

  t1
  |> tensor_to_list
  |> should.equal([[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]] |> dynamic.from)

  let t2 =
    build_tensor([4, 3, 2], fn(idx) {
      let assert [a, b, c] = idx
      { 6 * a + 2 * b + c } |> int.to_float
    })

  t2
  |> tensor_to_list
  |> should.equal(
    [
      [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
      [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
      [[12.0, 13.0], [14.0, 15.0], [16.0, 17.0]],
      [[18.0, 19.0], [20.0, 21.0], [22.0, 23.0]],
    ]
    |> dynamic.from,
  )

  shape(t1) |> should.equal([1, 2, 3])
  shape(t2) |> should.equal([4, 3, 2])

  shape(t1) |> size_of |> should.equal(6)
  shape(t2) |> size_of |> should.equal(24)

  rank(t1) |> should.equal(3)
  of_rank(3, t1) |> should.equal(True)
}

pub fn build_tensor_from_tensors_test() {
  build_tensor_from_tensors([tensor([1, 2, 3] |> dynamic.from)], fn(items) {
    let assert [item] = items
    let index = item.0
    let assert ScalarTensor(s) = item.1
    { int.to_float(index) +. s.real } |> to_scalar |> ScalarTensor
  })
  |> tensor_to_list
  |> should.equal([1.0, 3.0, 5.0] |> dynamic.from)

  build_tensor_from_tensors(
    [[1, 2, 3] |> dynamic.from |> tensor, [4, 5, 6] |> dynamic.from |> tensor],
    fn(items) {
      let assert [item1, item2] = items
      tensor_minus(item1.1, item2.1)
    },
  )
  |> tensor_to_list
  |> should.equal(
    [[-3.0, -4.0, -5.0], [-2.0, -3.0, -4.0], [-1.0, -2.0, -3.0]]
    |> dynamic.from,
  )
}

pub fn extend_opeartion_test() {
  let r0_td = 3.0 |> dynamic.from |> tensor
  let r1_td = [3.0, 4.0, 5.0] |> dynamic.from |> tensor
  let r2_td = [[3.0, 4.0, 5.0], [7.0, 8.0, 9.0]] |> dynamic.from |> tensor

  let assert ScalarTensor(s0) = r0_td
  s0.real |> should.equal(3.0)

  r1_td
  |> tensor1_map(fn(x) {
    let assert ScalarTensor(v) = x
    ScalarTensor(to_scalar(v.real +. 1.0))
  })
  |> tensor_to_list
  |> should.equal([4.0, 5.0, 6.0] |> dynamic.from)

  r2_td
  |> tensor_to_list
  |> should.equal(
    [[3.0, 4.0, 5.0], [7.0, 8.0, 9.0]]
    |> dynamic.from,
  )

  of_ranks(1, r1_td, 2, r2_td) |> should.be_true
  of_ranks(2, r1_td, 2, r2_td) |> should.be_false
  of_rank(1, r1_td) |> should.be_true
  of_rank(2, r2_td) |> should.be_true
}

pub fn classify_test() {
  dynamic.from(10) |> dynamic.classify |> should.equal("Int")
  dynamic.from(1.0) |> dynamic.classify |> should.equal("Float")
  dynamic.from([1.0]) |> dynamic.classify |> should.equal("List")
}

pub fn desc_test() {
  let t = [0.0, 1.0, 2.0, 3.0] |> dynamic.from |> tensor
  let u = [4.0, 5.0, 6.0, 7.0] |> dynamic.from |> tensor
  let add_0_0 =
    fn(ta, tb) {
      let assert ScalarTensor(a) = ta
      let assert ScalarTensor(b) = tb
      { a.real +. b.real } |> to_scalar |> ScalarTensor
    }
    |> ext2(0, 0)

  desc_u(add_0_0, t, u)
  |> tensor_to_list
  |> should.equal(
    [
      [4.0, 5.0, 6.0, 7.0],
      [5.0, 6.0, 7.0, 8.0],
      [6.0, 7.0, 8.0, 9.0],
      [7.0, 8.0, 9.0, 10.0],
    ]
    |> dynamic.from,
  )

  desc_t(add_0_0, u, t)
  |> tensor_to_list
  |> should.equal(
    [
      [4.0, 5.0, 6.0, 7.0],
      [5.0, 6.0, 7.0, 8.0],
      [6.0, 7.0, 8.0, 9.0],
      [7.0, 8.0, 9.0, 10.0],
    ]
    |> dynamic.from,
  )

  let sqr_0 =
    fn(t) {
      let assert ScalarTensor(a) = t
      let assert Ok(r) = float.power(a.real, 2.0)
      r |> to_scalar |> ScalarTensor
    }
    |> ext1(0)

  sqr_0(t)
  |> tensor_to_list
  |> should.equal([0.0, 1.0, 4.0, 9.0] |> dynamic.from)
}

pub fn map_to_scalar_test() {
  let t = tensor([0.0, 1.0, 2.0, 3.0] |> dynamic.from)

  fn(s: Scalar) { Scalar(s.id, s.real +. 1.0, s.link) }
  |> map_to_scalar(t)
  |> tensor_to_list
  |> should.equal([1.0, 2.0, 3.0, 4.0] |> dynamic.from)
}

pub fn auto_diff_test() {
  let s0 = to_scalar(0.0)
  let s1 = to_scalar(1.0)

  s0.real |> should.equal(0.0)

  let t0 = ScalarTensor(s0)
  let t1 = ScalarTensor(s1)

  gradient_once(t1, [t0, t1] |> ListTensor)
  |> tensor_to_list
  |> should.equal([0.0, 1.0] |> dynamic.from)
}

pub fn tensor_equal_test() {
  let t0 =
    tensor(
      [
        [
          [0.0, 2.0, 4.0, 6.0],
          [8.0, 10.0, 12.0, 14.0],
          [16.0, 18.0, 20.0, 22.0],
        ],
        [
          [24.0, 26.0, 28.0, 30.0],
          [32.0, 34.0, 36.0, 38.0],
          [40.0, 42.0, 44.0, 46.0],
        ],
      ]
      |> dynamic.from,
    )
  let t1 =
    tensor(
      [
        [
          [0.0, 2.00001, 4.00001, 6.00001],
          [8.00001, 10.00001, 12.00001, 14.00001],
          [16.00001, 18.00001, 20.00001, 22.00001],
        ],
        [
          [24.00001, 26.00001, 28.00001, 30.00001],
          [32.00001, 34.00001, 36.00001, 38.00001],
          [40.00001, 42.00001, 44.00001, 46.00001],
        ],
      ]
      |> dynamic.from,
    )
  let t2 =
    tensor(
      [
        [
          [
            [0.0, 2.00001, 4.00001, 6.00001],
            [8.00001, 10.00001, 12.00001, 14.00001],
            [16.00001, 18.00001, 20.00001, 22.00001],
          ],
          [
            [24.00001, 26.00001, 28.00001, 30.00001],
            [32.00001, 34.00001, 36.00001, 38.00001],
            [40.00001, 42.00001, 44.00001, 46.00001],
          ],
        ],
      ]
      |> dynamic.from,
    )

  is_tensor_equal(t0, t1) |> should.be_true
  is_tensor_equal(t0, t2) |> should.be_false
}

pub fn tensor_multiply_2_1_test() {
  let test_helper = tensor_multiply_2_1 |> check_theta_and_gradient2

  test_helper(
    tensor([[3, 4, 5, 6], [7, 8, 9, 10]] |> dynamic.from),
    tensor(
      [2, 3, 4, 5]
      |> dynamic.from,
    ),
    tensor(
      [[6.0, 12.0, 20.0, 30.0], [14.0, 24.0, 36.0, 50.0]]
      |> dynamic.from,
    ),
    "[[[2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0]], [10.0, 12.0, 14.0, 16.0]]",
  )

  test_helper(
    tensor([[1, 2], [3, 4]] |> dynamic.from),
    tensor([1, 0] |> dynamic.from),
    tensor([[1.0, 0.0], [3.0, 0.0]] |> dynamic.from),
    "[[[1.0, 0.0], [1.0, 0.0]], [4.0, 6.0]]",
  )

  test_helper(
    tensor(
      [[3, 4, 5, 6], [7, 8, 9, 10]]
      |> dynamic.from,
    ),
    tensor([[2, 3, 4, 5], [12, 13, 14, 15]] |> dynamic.from),
    tensor(
      [
        [[6.0, 12.0, 20.0, 30.0], [14.0, 24.0, 36.0, 50.0]],
        [[36.0, 52.0, 70.0, 90.0], [84.0, 104.0, 126.0, 150.0]],
      ]
      |> dynamic.from,
    ),
    "[[[14.0, 16.0, 18.0, 20.0], [14.0, 16.0, 18.0, 20.0]], [[10.0, 12.0, 14.0, 16.0], [10.0, 12.0, 14.0, 16.0]]]",
  )
}

pub fn tensor_sum_1_test() {
  let test_sum_helper = tensor_sum |> check_theta_and_gradient1

  test_sum_helper(
    tensor([3, 4, 5] |> dynamic.from),
    tensor(12 |> dynamic.from),
    tensor([1.0, 1.0, 1.0] |> dynamic.from),
  )
}

pub fn tensor_sum_2_test() {
  let a =
    tensor(
      [[3, 4, 5], [6, 7, 8]]
      |> dynamic.from,
    )

  is_tensor_equal(a, tensor([12, 21] |> dynamic.from))

  gradient_of(fn(b) { tensor_multiply(b, b) |> tensor_sum }, a)
  |> is_tensor_equal(
    [[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]] |> dynamic.from |> tensor,
  )
  |> should.be_true
}

pub fn tensor_sum_3_test() {
  let dot_product = fn(a, b) { tensor_multiply_2_1(a, b) |> tensor_sum }
  let sse = fn(a, b) { tensor_minus(a, b) |> tensor_sqr |> tensor_sum }

  let test_dot_product = dot_product |> check_theta_and_gradient2
  let test_sse = sse |> check_theta_and_gradient2
  {
    let a = [[3, 4, 5, 6], [7, 8, 9, 10]] |> dynamic.from |> tensor
    let b = [2, 3, 4, 5] |> dynamic.from |> tensor

    test_dot_product(
      a,
      b,
      tensor([68, 124] |> dynamic.from),
      "[[[2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0]], [10.0, 12.0, 14.0, 16.0]]",
    )

    test_sse(
      a,
      b,
      tensor([4, 100] |> dynamic.from),
      "[[[2.0, 2.0, 2.0, 2.0], [10.0, 10.0, 10.0, 10.0]], [-12.0, -12.0, -12.0, -12.0]]",
    )
  }

  test_dot_product(
    tensor(
      [[3, 4, 5, 6], [7, 8, 9, 10]]
      |> dynamic.from,
    ),
    tensor(
      [[2, 3, 4, 5], [12, 13, 14, 15]]
      |> dynamic.from,
    ),
    tensor(
      [[68, 124], [248, 464]]
      |> dynamic.from,
    ),
    "[[[14.0, 16.0, 18.0, 20.0], [14.0, 16.0, 18.0, 20.0]], [[10.0, 12.0, 14.0, 16.0], [10.0, 12.0, 14.0, 16.0]]]",
  )

  {
    let a = [[3, 4, 5], [6, 7, 8]] |> dynamic.from |> tensor

    tensor_sum_cols(a)
    |> is_tensor_equal([9, 11, 13] |> dynamic.from |> tensor)
    |> should.be_true

    gradient_of(fn(b) { tensor_multiply(b, b) |> tensor_sum_cols }, a)
    |> is_tensor_equal([[6, 8, 10], [12, 14, 16]] |> dynamic.from |> tensor)
    |> should.be_true
  }
}

pub fn argmax_test() {
  let test_helper = tensor_argmax |> check_theta_and_gradient1

  test_helper(
    tensor([0, 0, 1, 0] |> dynamic.from),
    tensor(2 |> dynamic.from),
    tensor([0, 0, 0, 0] |> dynamic.from),
  )

  test_helper(
    tensor(
      [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
      |> dynamic.from,
    ),
    tensor([2, 1, 0, 3] |> dynamic.from),
    tensor(
      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
      |> dynamic.from,
    ),
  )
}

pub fn max_test() {
  let y =
    [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    |> dynamic.from
    |> tensor

  check_theta_and_gradient1(tensor_max)(
    y,
    tensor([1, 1, 1, 1] |> dynamic.from),
    y,
  )
}

pub fn dot_product_test() {
  let signal =
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    |> dynamic.from
    |> tensor

  let filter_bank =
    [
      [[1, 2], [3, 4], [5, 6]],
      [[7, 8], [9, 10], [11, 12]],
      [[13, 14], [15, 16], [17, 18]],
      [[19, 20], [21, 22], [23, 24]],
    ]
    |> dynamic.from
    |> tensor

  // for testing b = 4
  //             m = 3
  //             d = 2

  dotted_product(
    tensor([1, 2, 3, 4] |> dynamic.from),
    tensor([1, 2, 3, 4] |> dynamic.from),
    0.0 |> to_scalar,
  )
  |> get_real
  |> should.equal(30.0)

  dot_product(
    tensor([1, 2, 3, 4] |> dynamic.from),
    tensor([2, 3, 4, 5] |> dynamic.from),
  )
  |> get_real
  |> should.equal(40.0)

  let bank0 = [[1, 2], [3, 4], [5, 6]] |> dynamic.from |> tensor
  let bank1 = [[7, 8], [9, 10], [11, 12]] |> dynamic.from |> tensor

  sum_dp(bank0, signal, -1, 0.0) |> get_real |> should.equal(50.0)
  sum_dp(bank1, signal, -1, 0.0) |> get_real |> should.equal(110.0)

  correlation_overlap(bank0, signal, 0) |> get_real |> should.equal(50.0)

  tensor_correlate(filter_bank, signal)
  |> tensor_to_list
  |> should.equal(
    [
      [50.0, 110.0, 170.0, 230.0],
      [91.0, 217.0, 343.0, 469.0],
      [133.0, 331.0, 529.0, 727.0],
      [175.0, 445.0, 715.0, 985.0],
      [217.0, 559.0, 901.0, 1243.0],
      [110.0, 362.0, 614.0, 866.0],
    ]
    |> dynamic.from,
  )

  let gs =
    fn(a, b) {
      gradient_of(
        fn(t) {
          let assert ListTensor([a, b]) = t
          tensor_correlate(a, b)
        },
        ListTensor([a, b]),
      )
    }(filter_bank, signal)

  let assert ListTensor([gs_1, gs_2, ..]) = gs
  gs_1
  |> tensor_to_list
  |> should.equal(
    [
      [[25.0, 30.0], [36.0, 42.0], [35.0, 40.0]],
      [[25.0, 30.0], [36.0, 42.0], [35.0, 40.0]],
      [[25.0, 30.0], [36.0, 42.0], [35.0, 40.0]],
      [[25.0, 30.0], [36.0, 42.0], [35.0, 40.0]],
    ]
    |> dynamic.from,
  )

  gs_2
  |> tensor_to_list
  |> should.equal(
    [
      [88.0, 96.0],
      [144.0, 156.0],
      [144.0, 156.0],
      [144.0, 156.0],
      [144.0, 156.0],
      [104.0, 112.0],
    ]
    |> dynamic.from,
  )
}

fn lift_float2(f: fn(Scalar, Scalar) -> Scalar) -> fn(Float, Float) -> Float {
  fn(x, y) {
    f(x |> to_scalar, y |> to_scalar)
    |> get_real
  }
}

fn lift_float1(f: fn(Scalar) -> Scalar) -> fn(Float) -> Float {
  fn(x) { x |> to_scalar |> f |> get_real }
}

pub fn recify_test() {
  { rectify_0 |> lift_float1 }(3.0) |> should.equal(3.0)
  { rectify_0 |> lift_float1 }(-3.0) |> should.equal(0.0)

  { learner.add_0_0() |> lift_float2 }(0.0, -3.0)
  |> { rectify_0 |> lift_float1 }
  |> should.equal(0.0)

  { learner.multiply_0_0() |> lift_float2 }(1.0, -3.0)
  |> { rectify_0 |> lift_float1 }
  |> should.equal(0.0)

  { learner.add_0_0() |> lift_float2 }(0.0, 3.0)
  |> { rectify_0 |> lift_float1 }
  |> should.equal(3.0)

  { learner.multiply_0_0() |> lift_float2 }(1.0, 3.0)
  |> { rectify_0 |> lift_float1 }
  |> should.equal(3.0)

  tensor(
    [1.0, 2.3, -1.1]
    |> dynamic.from,
  )
  |> rectify
  |> tensor_to_list
  |> should.equal(
    [1.0, 2.3, 0.0]
    |> dynamic.from,
  )

  learner.tensor_add(
    tensor([1.0, 2.3, -1.1] |> dynamic.from),
    tensor([1.0, 2.3, -1.1] |> dynamic.from),
  )
  |> rectify
  |> tensor_to_list
  |> should.equal([2.0, 4.6, 0.0] |> dynamic.from)
}
