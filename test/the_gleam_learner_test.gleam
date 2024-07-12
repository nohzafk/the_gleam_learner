import gleam/dict
import gleam/dynamic.{type Dynamic}
import gleam/float
import gleam/int
import gleam/io
import gleam/result
import gleam/string
import gleeunit
import gleeunit/should

import learner.{
  type Scalar, ListTensor, Scalar, ScalarTensor, build_tensor,
  build_tensor_from_tensors, correlation_overlap, desc_t, desc_u, dot_product,
  dotted_product, ext1, ext2, from_float, get_real, gradient_once,
  gradient_operator, is_tensor_equal, map_to_scalar, of_rank, of_ranks, rank,
  rectify, rectify_0, shape, size_of, sum_dp, tensor, tensor1_map, tensor_argmax,
  tensor_correlate, tensor_max, tensor_minus, tensor_multiply,
  tensor_multiply_2_1, tensor_sqr, tensor_sum, tensor_sum_cols, tensor_to_list,
}

pub fn main() {
  gleeunit.main()
}

pub fn dual_as_key_test() {
  let d1 = from_float(0.1)
  let v1 = 1.0
  let sigma = dict.new() |> dict.insert(d1, v1)

  sigma |> dict.size() |> should.equal(1)
  sigma |> dict.get(d1) |> result.unwrap(0.0) |> should.equal(v1)

  let d2 = from_float(0.1)
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
    { int.to_float(index) +. s.real } |> from_float |> ScalarTensor
  })
  |> tensor_to_list
  |> should.equal([1.0, 3.0, 5.0] |> dynamic.from)

  build_tensor_from_tensors(
    [tensor([1, 2, 3] |> dynamic.from), tensor([4, 5, 6] |> dynamic.from)],
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
  let r0_td = tensor(3.0 |> dynamic.from)
  let r1_td = tensor([3.0, 4.0, 5.0] |> dynamic.from)
  let r2_td =
    tensor(
      [[3.0, 4.0, 5.0], [7.0, 8.0, 9.0]]
      |> dynamic.from,
    )

  let assert ScalarTensor(s0) = r0_td
  s0.real |> should.equal(3.0)

  r1_td
  |> tensor1_map(fn(x) {
    let assert ScalarTensor(v) = x
    ScalarTensor(from_float(v.real +. 1.0))
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
  dynamic.from([1.0]) |> dynamic.classify |> should.equal("List")
}

pub fn desc_test() {
  let t = tensor([0.0, 1.0, 2.0, 3.0] |> dynamic.from)
  let u = tensor([4.0, 5.0, 6.0, 7.0] |> dynamic.from)

  let add_0_0 =
    fn(ta, tb) {
      let assert ScalarTensor(a) = ta
      let assert ScalarTensor(b) = tb
      { a.real +. b.real } |> from_float |> ScalarTensor
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
      r |> from_float |> ScalarTensor
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
  let s0 = from_float(0.0)
  let s1 = from_float(1.0)

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

fn test_helper1(f) {
  fn(t, answers, gradients) {
    should.equal(f(t) |> tensor_to_list, answers |> tensor_to_list)
    should.equal(
      gradient_operator(f, t) |> tensor_to_list,
      gradients |> tensor_to_list,
    )
  }
}

fn test_helper2(f) {
  fn(a, b, answers, gradients: String) {
    should.equal(f(a, b) |> tensor_to_list, answers |> tensor_to_list)

    gradient_operator(
      fn(t) {
        let assert ListTensor([a, b]) = t
        f(a, b)
      },
      ListTensor([a, b]),
    )
    |> tensor_to_list
    |> string.inspect
    |> should.equal(gradients)
  }
}

pub fn tensor_multiply_2_1_test() {
  let test_helper = tensor_multiply_2_1 |> test_helper2

  test_helper(
    tensor(
      [[3, 4, 5, 6], [7, 8, 9, 10]]
      |> dynamic.from,
    ),
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
  let test_sum_helper = tensor_sum |> test_helper1

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

  gradient_operator(fn(b) { tensor_multiply(b, b) |> tensor_sum }, a)
  |> is_tensor_equal(tensor(
    [[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]
    |> dynamic.from,
  ))
  |> should.be_true
}

pub fn tensor_sum_3_test() {
  let dot_product = fn(a, b) { tensor_multiply_2_1(a, b) |> tensor_sum }
  let sse = fn(a, b) { tensor_minus(a, b) |> tensor_sqr |> tensor_sum }

  let test_dot_product = dot_product |> test_helper2
  let test_sse = sse |> test_helper2
  {
    let a =
      tensor(
        [[3, 4, 5, 6], [7, 8, 9, 10]]
        |> dynamic.from,
      )
    let b =
      tensor(
        [2, 3, 4, 5]
        |> dynamic.from,
      )

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
    let a = tensor([[3, 4, 5], [6, 7, 8]] |> dynamic.from)

    tensor_sum_cols(a)
    |> is_tensor_equal(tensor([9, 11, 13] |> dynamic.from))
    |> should.be_true

    gradient_operator(fn(b) { tensor_multiply(b, b) |> tensor_sum_cols }, a)
    |> is_tensor_equal(tensor(
      [[6, 8, 10], [12, 14, 16]]
      |> dynamic.from,
    ))
    |> should.be_true
  }
}

pub fn argmax_test() {
  let test_helper = tensor_argmax |> test_helper1

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
    tensor(
      [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
      |> dynamic.from,
    )

  test_helper1(tensor_max)(y, tensor([1, 1, 1, 1] |> dynamic.from), y)
}

pub fn dot_product_test() {
  let signal =
    tensor(
      [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
      |> dynamic.from,
    )

  let filter_bank =
    tensor(
      [
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]],
        [[13, 14], [15, 16], [17, 18]],
        [[19, 20], [21, 22], [23, 24]],
      ]
      |> dynamic.from,
    )

  // for testing b = 4
  //             m = 3
  //             d = 2

  dotted_product(
    tensor([1, 2, 3, 4] |> dynamic.from),
    tensor([1, 2, 3, 4] |> dynamic.from),
    0.0 |> from_float,
  )
  |> get_real
  |> should.equal(30.0)

  dot_product(
    tensor([1, 2, 3, 4] |> dynamic.from),
    tensor([2, 3, 4, 5] |> dynamic.from),
  )
  |> get_real
  |> should.equal(40.0)

  let bank0 = tensor([[1, 2], [3, 4], [5, 6]] |> dynamic.from)
  let bank1 = tensor([[7, 8], [9, 10], [11, 12]] |> dynamic.from)

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
      gradient_operator(
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
    f(x |> from_float, y |> from_float)
    |> get_real
  }
}

fn lift_float1(f: fn(Scalar) -> Scalar) -> fn(Float) -> Float {
  fn(x) { x |> from_float |> f |> get_real }
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
