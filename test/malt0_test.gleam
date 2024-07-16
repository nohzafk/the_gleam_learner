import gleam/dict
import gleam/dynamic.{type Dynamic}
import gleam/float
import gleam/function
import gleam/int
import gleam/io
import gleam/list
import gleam/result
import gleam/string
import gleeunit/should
import malt0.{
  type Scalar, type Tensor, type Theta, ListTensor, Scalar, ScalarTensor,
  adam_gradient_descent, add_0_0, build_tensor, build_tensor_from_tensors,
  correlation_overlap, desc_t, desc_u, dot_product, dotted_product, ext1, ext2,
  get_float, get_real, get_scalar, gradient_descent, gradient_of, gradient_once,
  hp_new, hp_new_batch_size, hp_new_beta, hp_new_mu, make_theta, map_tensor,
  map_tensor_recursively, multiply_0_0, naked_gradient_descent, new_scalar, rank,
  rectify, rectify_0, relu, rms_gradient_descent, samples, sampling_obj, shape,
  smooth, sum_dp, tensor, tensor_add, tensor_argmax, tensor_correlate,
  tensor_divide, tensor_dot_product, tensor_dot_product_2_1, tensor_exp,
  tensor_log, tensor_max, tensor_minus, tensor_multiply, tensor_multiply_2_1,
  tensor_sqr, tensor_sqrt, tensor_sum, tensor_sum_cols, tensorized_cmp_equal,
  tlen, to_tensor, tolerace, velocity_gradient_descent, zeros,
}

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

pub fn is_tensor_equal(ta: Tensor, tb: Tensor) -> Bool {
  case ta, tb {
    ScalarTensor(Scalar(real: a, ..)), ScalarTensor(Scalar(real: b, ..)) ->
      float.loosely_equals(a, b, tolerace)
    ListTensor(a), ListTensor(b) ->
      case tlen(ta) == tlen(tb) {
        True ->
          list.map2(a, b, is_tensor_equal)
          |> list.all(fn(x) { x })
        _ -> False
      }
    _, _ -> False
  }
}

fn tensor_should_equal(actual, expected) {
  case is_tensor_equal(actual, expected) {
    True -> {
      // io.debug(actual |> tensor_to_list)
      should.be_true(True)
    }
    False -> should.equal(actual |> tensor_to_list, expected |> tensor_to_list)
  }
}

pub fn tlen_test() {
  let t0 = [1.0, 3.1, 2.9] |> dynamic.from |> tensor
  t0 |> tlen |> should.equal(3)
}

pub fn dual_as_key_test() {
  let d1 = new_scalar(0.1)
  let v1 = 1.0
  let sigma = dict.new() |> dict.insert(d1, v1)

  sigma |> dict.size() |> should.equal(1)
  sigma |> dict.get(d1) |> result.unwrap(0.0) |> should.equal(v1)

  let d2 = new_scalar(0.1)
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
  rank(t1) |> should.equal(3)
}

pub fn build_tensor_from_tensors_test() {
  build_tensor_from_tensors([tensor([1, 2, 3] |> dynamic.from)], fn(items) {
    let assert [item] = items
    let index = item.0
    let assert ScalarTensor(s) = item.1
    { int.to_float(index) +. s.real } |> new_scalar |> ScalarTensor
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
  |> map_tensor(fn(x) {
    let assert ScalarTensor(v) = x
    ScalarTensor(new_scalar(v.real +. 1.0))
  })
  |> tensor_should_equal([4.0, 5.0, 6.0] |> dynamic.from |> tensor)

  r2_td
  |> tensor_to_list
  |> should.equal(
    [[3.0, 4.0, 5.0], [7.0, 8.0, 9.0]]
    |> dynamic.from,
  )
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
      { a.real +. b.real } |> new_scalar |> ScalarTensor
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
      r |> new_scalar |> ScalarTensor
    }
    |> ext1(0)

  sqr_0(t)
  |> tensor_to_list
  |> should.equal([0.0, 1.0, 4.0, 9.0] |> dynamic.from)
}

pub fn map_tensor_recursively_test() {
  let t = tensor([0.0, 1.0, 2.0, 3.0] |> dynamic.from)

  fn(s: Scalar) { Scalar(s.id, s.real +. 1.0, s.link) }
  |> map_tensor_recursively(t)
  |> tensor_to_list
  |> should.equal([1.0, 2.0, 3.0, 4.0] |> dynamic.from)
}

pub fn auto_diff_test() {
  let s0 = new_scalar(0.0)
  let s1 = new_scalar(1.0)

  s0.real |> should.equal(0.0)

  let t0 = ScalarTensor(s0)
  let t1 = ScalarTensor(s1)

  gradient_once(t1, [t0, t1])
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

pub fn check_theta_and_gradient1(f) {
  fn(t, answers, gradients) {
    tensor_should_equal(f(t), answers)

    let assert ListTensor(lst) = t
    let f_wrapper = fn(lst) { f(lst |> ListTensor) }
    tensor_should_equal(gradient_of(f_wrapper, lst) |> ListTensor, gradients)
  }
}

pub fn check_theta_and_gradient(f) {
  fn(t, u, answers, gradients) {
    is_tensor_equal(f(t, u), answers) |> should.be_true

    let f_wrapper = fn(lst) {
      let assert [a, b] = lst
      f(a, b)
    }

    gradient_of(f_wrapper, [t, u])
    |> list.map(tensor_to_list)
    |> string.inspect
    |> should.equal(gradients)
  }
}

pub fn tensor_multiply_2_1_test() {
  let test_helper = tensor_multiply_2_1 |> check_theta_and_gradient

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
  let a = [[3, 4, 5], [6, 7, 8]] |> dynamic.from |> tensor

  tensor_should_equal(tensor_sum(a), [12, 21] |> dynamic.from |> tensor)

  fn(lst) {
    let assert [b] = lst
    tensor_multiply(b, b) |> tensor_sum
  }
  |> gradient_of([a])
  |> ListTensor
  |> tensor_should_equal(
    [[[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]] |> dynamic.from |> tensor,
  )
}

pub fn tensor_sum_3_test() {
  let dot_product = fn(a, b) { tensor_multiply_2_1(a, b) |> tensor_sum }
  let sse = fn(a, b) { tensor_minus(a, b) |> tensor_sqr |> tensor_sum }

  let test_dot_product = dot_product |> check_theta_and_gradient
  let test_sse = sse |> check_theta_and_gradient
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
    |> tensor_should_equal([9, 11, 13] |> dynamic.from |> tensor)

    fn(lst) {
      let assert [b] = lst
      tensor_multiply(b, b) |> tensor_sum
    }
    |> gradient_of([a])
    |> ListTensor
    |> tensor_should_equal(
      [[[6, 8, 10], [12, 14, 16]]] |> dynamic.from |> tensor,
    )
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

pub fn correlate_test() {
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
    to_tensor(0.0),
  )
  |> tensor_should_equal(30.0 |> dynamic.from |> tensor)

  dot_product(
    tensor([1, 2, 3, 4] |> dynamic.from),
    tensor([2, 3, 4, 5] |> dynamic.from),
  )
  |> tensor_should_equal(40.0 |> dynamic.from |> tensor)

  let bank0 = [[1, 2], [3, 4], [5, 6]] |> dynamic.from |> tensor
  let bank1 = [[7, 8], [9, 10], [11, 12]] |> dynamic.from |> tensor

  sum_dp(bank0, signal, -1, 0.0) |> get_real |> should.equal(50.0)
  sum_dp(bank1, signal, -1, 0.0) |> get_real |> should.equal(110.0)

  correlation_overlap(bank0, signal, 0) |> get_real |> should.equal(50.0)

  tensor_correlate(filter_bank, signal)
  |> tensor_should_equal(
    [
      [50.0, 110.0, 170.0, 230.0],
      [91.0, 217.0, 343.0, 469.0],
      [133.0, 331.0, 529.0, 727.0],
      [175.0, 445.0, 715.0, 985.0],
      [217.0, 559.0, 901.0, 1243.0],
      [110.0, 362.0, 614.0, 866.0],
    ]
    |> dynamic.from
    |> tensor,
  )

  let gs =
    fn(a, b) {
      gradient_of(
        fn(t) {
          let assert [a, b] = t
          tensor_correlate(a, b)
        },
        [a, b],
      )
    }(filter_bank, signal)

  let assert [gs_1, gs_2, ..] = gs
  tensor_should_equal(
    gs_1,
    [
      [[25.0, 30.0], [36.0, 42.0], [35.0, 40.0]],
      [[25.0, 30.0], [36.0, 42.0], [35.0, 40.0]],
      [[25.0, 30.0], [36.0, 42.0], [35.0, 40.0]],
      [[25.0, 30.0], [36.0, 42.0], [35.0, 40.0]],
    ]
      |> dynamic.from
      |> tensor,
  )

  tensor_should_equal(
    gs_2,
    [
      [88.0, 96.0],
      [144.0, 156.0],
      [144.0, 156.0],
      [144.0, 156.0],
      [144.0, 156.0],
      [104.0, 112.0],
    ]
      |> dynamic.from
      |> tensor,
  )
}

pub fn lower_to_float1(f: fn(Tensor) -> Tensor) -> fn(Float) -> Float {
  fn(v) {
    let assert ScalarTensor(s) = v |> to_tensor |> f
    s.real
  }
}

pub fn lower_to_float2(
  f: fn(Tensor, Tensor) -> Tensor,
) -> fn(Float, Float) -> Float {
  fn(a, b) {
    let assert ScalarTensor(s) = f(a |> to_tensor, b |> to_tensor)
    s.real
  }
}

fn tensor_op_wrapper(f) {
  fn(theta) {
    let assert [a, b] = theta
    f(a, b)
  }
}

fn tensor_op_wrapper1(f) {
  fn(theta) {
    let assert [a] = theta
    f(a)
  }
}

pub fn a_core_test() {
  {
    let a = 7.0
    let b = 13.0
    let theta = [a, b] |> list.map(fn(x) { x |> dynamic.from |> tensor })

    { tensor_add |> lower_to_float2 }(a, b)
    |> should.equal(20.0)

    { tensor_add |> tensor_op_wrapper }
    |> gradient_of(theta)
    |> ListTensor
    |> tensor_should_equal([1.0, 1.0] |> dynamic.from |> tensor)

    { tensor_multiply |> lower_to_float2 }(a, b)
    |> should.equal(91.0)

    { tensor_multiply |> tensor_op_wrapper }
    |> gradient_of(theta)
    |> ListTensor
    |> tensor_should_equal([13.0, 7.0] |> dynamic.from |> tensor)

    { tensor_divide |> tensor_op_wrapper }
    |> gradient_of(theta)
    |> ListTensor
    |> tensor_should_equal([0.07692, -0.04142] |> dynamic.from |> tensor)
  }

  {
    let a = [7, 8, 9] |> dynamic.from |> tensor

    tensor_exp(a)
    |> tensor_should_equal(
      [1096.6331, 2980.9579, 8103.0839] |> dynamic.from |> tensor,
    )

    { tensor_exp |> tensor_op_wrapper1 }
    |> gradient_of([a])
    |> ListTensor
    |> tensor_should_equal(
      [[1096.6331, 2980.9579, 8103.0839]] |> dynamic.from |> tensor,
    )

    tensor_log(a)
    |> tensor_should_equal([1.9459, 2.0794, 2.1972] |> dynamic.from |> tensor)

    { tensor_log |> tensor_op_wrapper1 }
    |> gradient_of([a])
    |> ListTensor
    |> tensor_should_equal([[0.1428, 0.125, 0.1111]] |> dynamic.from |> tensor)

    tensor_sqrt(a)
    |> tensor_should_equal([2.6457, 2.8284, 3.0] |> dynamic.from |> tensor)
  }

  {
    let t2 = [[3, 4, 5], [7, 8, 9]] |> dynamic.from |> tensor
    let t1 = [4, 5, 6] |> dynamic.from |> tensor

    tensor_dot_product(t1, t1)
    |> tensor_should_equal(77.0 |> dynamic.from |> tensor)

    tensor_multiply_2_1(
      t2,
      [[4, 5, 6], [4, 5, 6], [4, 5, 6]]
        |> dynamic.from
        |> tensor,
    )
    |> tensor_should_equal(
      [
        [[12, 20, 30], [28, 40, 54]],
        [[12, 20, 30], [28, 40, 54]],
        [[12, 20, 30], [28, 40, 54]],
      ]
      |> dynamic.from
      |> tensor,
    )

    tensor_dot_product_2_1(t2, t1)
    |> tensor_should_equal([62, 122] |> dynamic.from |> tensor)

    tensor_sqr(t2)
    |> tensor_should_equal(
      [[9, 16, 25], [49, 64, 81]]
      |> dynamic.from
      |> tensor,
    )
  }
}

pub fn d_gradient_descent_test() {
  let obj = fn(theta: Theta) {
    let assert [a, ..] = theta
    to_tensor(30.0) |> tensor_minus(a) |> tensor_sqr
  }

  let id = function.identity

  let hp = hp_new(revs: 500, alpha: 0.01)

  let naked_gd =
    { hp |> gradient_descent }(id, id, fn(hp) {
      fn(w, g) {
        tensor_multiply(g, hp.alpha |> to_tensor)
        |> tensor_minus(w, _)
      }
    })

  naked_gd(obj, [3.0 |> dynamic.from |> tensor])
  |> ListTensor
  |> tensor_should_equal([29.998892352401082] |> dynamic.from |> tensor)
}

pub fn e_gd_common_test() {
  zeros([1, 2, 3] |> dynamic.from |> tensor)
  |> tensor_should_equal([0, 0, 0] |> dynamic.from |> tensor)

  smooth(0.9, 31.0, -8.0) |> should.equal(27.1)
}

pub fn f_naked_test() {
  let obj = fn(theta) {
    let assert [a] = theta
    tensor_minus(to_tensor(30.0), a) |> tensor_sqr
  }

  let hp = hp_new(revs: 400, alpha: 0.01)
  { hp |> naked_gradient_descent }(obj, [3.0 |> dynamic.from |> tensor])
  |> ListTensor
  |> tensor_should_equal([29.991647931623252] |> dynamic.from |> tensor)
}

pub fn g_velocity_gradient_descent() {
  let obj = fn(theta) {
    let assert [a] = theta
    tensor_minus(to_tensor(30.0), a) |> tensor_sqr
  }

  let hp = hp_new(revs: 70, alpha: 0.01) |> hp_new_mu(mu: 0.9)
  { hp |> velocity_gradient_descent }(obj, [3.0 |> dynamic.from |> tensor])
  |> ListTensor
  |> tensor_should_equal([30.686162582787535] |> dynamic.from |> tensor)
}

pub fn h_rms_gradient_descent_test() {
  let obj = fn(theta) {
    let assert [a] = theta
    tensor_minus(to_tensor(30.0), a) |> tensor_sqr
  }

  let hp = hp_new(revs: 170, alpha: 0.1) |> hp_new_beta(beta: 0.999)
  { hp |> rms_gradient_descent }(obj, [3.0 |> dynamic.from |> tensor])
  |> ListTensor
  |> tensor_should_equal([29.990436454407956] |> dynamic.from |> tensor)
}

pub fn i_adam_test() {
  let obj = fn(theta) {
    let assert [a] = theta
    tensor_minus(to_tensor(30.0), a) |> tensor_sqr
  }
  let hp =
    hp_new(revs: 150, alpha: 0.1)
    |> hp_new_beta(beta: 0.999)
    |> hp_new_mu(mu: 0.9)
  { hp |> adam_gradient_descent }(obj, [3.0 |> dynamic.from |> tensor])
  |> ListTensor
  |> tensor_should_equal([29.994907156105718] |> dynamic.from |> tensor)
}

pub fn j_stochastic_test() {
  let test_samples = samples(10, 3)

  test_samples |> list.length |> should.equal(3)

  test_samples
  |> list.all(fn(x) { x >= 0 && x <= 9 })
  |> should.be_true

  let test_expectant_fn = fn(xs, ys) {
    fn(theta) {
      should.equal(shape(xs), shape(ys))

      tensorized_cmp_equal(xs, ys)
      |> tensor_sum
      |> get_float
      |> should.equal(3.0)

      theta
    }
  }

  let test_tensor = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] |> dynamic.from |> tensor

  let hp = hp_new(1, 0.1) |> hp_new_batch_size(batch_size: 3)
  { hp |> sampling_obj }(test_expectant_fn, test_tensor, test_tensor)(False)
}

pub fn recify_test() {
  rectify_0(new_scalar(3.0)) |> get_real |> should.equal(3.0)
  rectify_0(new_scalar(-3.0)) |> get_real |> should.equal(0.0)

  let lower_tensor_op = fn(f: fn(Tensor, Tensor) -> Tensor) -> fn(Float, Float) ->
    Scalar {
    fn(a: Float, b: Float) { f(a |> to_tensor, b |> to_tensor) |> get_scalar }
  }

  { add_0_0() |> lower_tensor_op }(0.0, -3.0)
  |> rectify_0
  |> get_real
  |> should.equal(0.0)

  { multiply_0_0() |> lower_tensor_op }(1.0, -3.0)
  |> rectify_0
  |> get_real
  |> should.equal(0.0)

  { add_0_0() |> lower_tensor_op }(0.0, 3.0)
  |> rectify_0
  |> get_real
  |> should.equal(3.0)

  { multiply_0_0() |> lower_tensor_op }(1.0, 3.0)
  |> rectify_0
  |> get_real
  |> should.equal(3.0)

  rectify([1.0, 2.3, -1.1] |> dynamic.from |> tensor)
  |> tensor_should_equal([1.0, 2.3, 0.0] |> dynamic.from |> tensor)

  tensor_add(
    tensor([1.0, 2.3, -1.1] |> dynamic.from),
    tensor([1.0, 2.3, -1.1] |> dynamic.from),
  )
  |> rectify
  |> tensor_should_equal([2.0, 4.6, 0.0] |> dynamic.from |> tensor)
}

pub fn k_dense_test() {
  gradient_of(
    relu([-1, 0, 1] |> dynamic.from |> tensor),
    make_theta([[1, 2, 3], [1, 2, 3]] |> dynamic.from, 2.0),
  )
  |> ListTensor
  |> tensor_should_equal(
    make_theta([[-1, 0, 1], [-1, 0, 1]] |> dynamic.from, 2.0)
    |> ListTensor,
  )
}
