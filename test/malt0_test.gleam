import gleam/dict
import gleam/dynamic.{type Dynamic}
import gleam/float
import gleam/function
import gleam/int
import gleam/iterator
import gleam/list
import gleam/result
import gleam/string
import gleam_community/maths/combinatorics
import gleeunit/should
import malt0.{
  type Scalar, type Tensor, type Theta, Block, ListTensor, Scalar, ScalarTensor,
  accuracy, adam_gradient_descent, add_0_0, build_tensor,
  build_tensor_from_tensors, compose_block_fns, corr, correlation_overlap,
  desc_t, desc_u, dot_product, dotted_product, ext1, ext2, get_float, get_real,
  get_scalar, gradient_descent, gradient_of, gradient_once, hp_new,
  hp_new_batch_size, hp_new_beta, hp_new_mu, init_shape, k_recu, l2_loss,
  make_theta, map_tensor, map_tensor_recursively, multiply_0_0,
  naked_gradient_descent, new_scalar, plane, rank, rectify, rectify_0, recu,
  relu, rms_gradient_descent, samples, sampling_obj, shape, smooth, stack2,
  stack_blocks, sum_dp, tensor, tensor_abs, tensor_add, tensor_argmax,
  tensor_correlate, tensor_divide, tensor_dot_product, tensor_dot_product_2_1,
  tensor_exp, tensor_log, tensor_max, tensor_minus, tensor_multiply,
  tensor_multiply_2_1, tensor_sqr, tensor_sqrt, tensor_sum, tensor_sum_cols,
  tensorized_cmp_equal, tlen, to_tensor, tolerace, velocity_gradient_descent,
  zeros,
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

pub fn check_gradients1(f, theta: Theta, gradients) {
  let f_wrapper = fn(lst) { f(lst |> ListTensor) }
  gradient_of(f_wrapper, theta) |> ListTensor |> tensor_should_equal(gradients)
}

pub fn check_theta_and_gradient1(f) {
  fn(t, answers, gradients) {
    tensor_should_equal(f(t), answers)

    let assert ListTensor(theta) = t
    check_gradients1(f, theta, gradients)
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

  { tensor_max |> check_theta_and_gradient1 }(
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

pub fn c_loss_test() {
  let r2d1 = [[3, 4, 5], [3, 4, 5]] |> dynamic.from |> tensor

  {
    fn(t) {
      let assert ListTensor(lst) = t
      lst |> l2_loss(plane)(r2d1, [1, 1] |> dynamic.from |> tensor)
    }
    |> check_theta_and_gradient1
  }(
    make_theta([0, 0, 0] |> dynamic.from, 0.0) |> ListTensor,
    2.0 |> dynamic.from |> tensor,
    make_theta([-12, -16, -20] |> dynamic.from, -4.0) |> ListTensor,
  )
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

  smooth(to_tensor(0.9), to_tensor(31.0), to_tensor(-8.0))
  |> get_float
  |> should.equal(27.1)
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
  { hp.batch_size |> sampling_obj }(test_expectant_fn, test_tensor, test_tensor)(
    False,
  )
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

  check_gradients1(
    fn(a) {
      let assert ListTensor(lst) = a
      lst |> relu([-1, 0, 1] |> dynamic.from |> tensor)
    },
    make_theta([[1, 2, -3], [1, 2, -3]] |> dynamic.from, 2.0),
    make_theta(
      [[0, 0, 0], [0, 0, 0]]
        |> dynamic.from,
      0.0,
    )
      |> ListTensor,
  )

  [
    [
      [1.5043209265388457, 1.5892702938568741],
      [0.33592431328374556, 0.8653082103400623],
      [-0.8007586188977664, -1.4723530725283407],
    ]
      |> dynamic.from
      |> tensor,
    [0, 0, 0] |> dynamic.from |> tensor,
  ]
  |> relu([[2, 3], [2, 3]] |> dynamic.from |> tensor)
  |> tensor_should_equal(
    [
      [7.7764527346483145, 3.267773257587678, 0.0],
      [7.7764527346483145, 3.267773257587678, 0.0],
    ]
    |> dynamic.from
    |> tensor,
  )

  make_theta([[1, 2, 3], [1, 2, 3]] |> dynamic.from, 2.0)
  |> relu([[-1, 0, 1], [-1, 0, 1]] |> dynamic.from |> tensor)
  |> tensor_should_equal([[4, 4], [4, 4]] |> dynamic.from |> tensor)

  gradient_of(
    relu([[-1, 0, 1], [-1, 0, 1]] |> dynamic.from |> tensor),
    make_theta([[1, 2, 3], [1, 2, 3]] |> dynamic.from, 2.0),
  )
  |> ListTensor
  |> tensor_should_equal(
    make_theta(
      [[-2, 0, 2], [-2, 0, 2]]
        |> dynamic.from,
      4.0,
    )
    |> ListTensor,
  )
}

pub fn n_blocks_test() {
  let bf1 = fn(t) {
    fn(theta) {
      let assert [a, ..] = theta
      tensor_multiply(a, t)
    }
  }

  let bf2 = fn(t) {
    fn(theta) {
      let assert [a, ..] = theta
      tensor_multiply(a, t)
    }
  }

  let bf1bf2 = compose_block_fns(bf1, bf2, 1)

  [2.0, 4.0]
  |> list.map(fn(x) { x |> dynamic.from |> tensor })
  |> bf1bf2(4.0 |> dynamic.from |> tensor)
  |> get_float
  |> should.equal(32.0)

  let b1 = Block(bf1, [[]])
  let b2 = Block(bf2, [[]])
  let b1b2 = stack2(b1, b2)

  [2.0, 4.0]
  |> list.map(fn(x) { x |> dynamic.from |> tensor })
  |> b1b2.block_fn(4.0 |> dynamic.from |> tensor)
  |> get_float
  |> should.equal(32.0)

  let b1b2_again = stack_blocks([b1, b2])

  [2.0, 4.0]
  |> list.map(fn(x) { x |> dynamic.from |> tensor })
  |> b1b2_again.block_fn(4.0 |> dynamic.from |> tensor)
  |> get_float
  |> should.equal(32.0)
}

pub fn o_init_test() {
  let v = init_shape([1000, 4])

  let mean_v =
    v
    |> tensor_sum
    |> tensor_sum
    |> tensor_divide(to_tensor(4000.0))
    |> tensor_abs

  let variance_v =
    tensor_multiply(v, v)
    |> tensor_sum
    |> tensor_sum
    |> tensor_divide(to_tensor(4000.0))
    |> tensor_minus(tensor_multiply(mean_v, mean_v))

  { mean_v |> get_float <. 0.05 } |> should.be_true
  {
    let variance_v_float = variance_v |> get_float
    variance_v_float >=. 0.4 && variance_v_float <=. 0.6
  }
  |> should.be_true

  // Here variance will be 2/8 = 0.25
  let r = init_shape([1000, 4, 2])
  let mean_r =
    r
    |> tensor_sum
    |> tensor_sum
    |> tensor_sum
    |> tensor_divide(to_tensor(8000.0))

  let variance_r =
    tensor_multiply(r, r)
    |> tensor_sum
    |> tensor_sum
    |> tensor_sum
    |> tensor_divide(to_tensor(8000.0))
    |> tensor_minus(tensor_multiply(mean_r, mean_r))

  { mean_r |> get_float <. 0.05 } |> should.be_true
  {
    let variance_r_float = variance_r |> get_float
    variance_r_float >=. 0.22 && variance_r_float <=. 0.28
  }
  |> should.be_true
}

pub fn l_accuracy_test() {
  let t2 = [[1, 2, 3, 4], [5, 6, 7, 8]] |> dynamic.from |> tensor

  let a_model = fn(t) { t }

  accuracy(a_model, t2, t2) |> should.equal(1.0)
}

pub fn m_recu_test() {
  let s2d1 =
    [[3], [4], [5], [6], [7], [8]]
    |> dynamic.from
    |> tensor

  let b2f3d1 =
    [[[3], [4], [5]], [[3], [4], [5]]]
    |> dynamic.from
    |> tensor

  let b2f3d2 =
    [[[3.0, 3.5], [4.0, 4.5], [5.0, 5.5]], [[3.0, 3.5], [4.0, 4.5], [5.0, 5.5]]]
    |> dynamic.from
    |> tensor

  let positive_biases = [3, 4] |> dynamic.from |> tensor

  [b2f3d1, positive_biases]
  |> corr(s2d1)
  |> tensor_should_equal(
    [[35, 36], [53, 54], [65, 66], [77, 78], [89, 90], [56, 57]]
    |> dynamic.from
    |> tensor,
  )

  [b2f3d1, positive_biases]
  |> recu(s2d1)
  |> tensor_should_equal(
    [[35, 36], [53, 54], [65, 66], [77, 78], [89, 90], [56, 57]]
    |> dynamic.from
    |> tensor,
  )

  let negative_biases = [-70, -60] |> dynamic.from |> tensor

  [b2f3d1, negative_biases]
  |> corr(s2d1)
  |> tensor_should_equal(
    [[-38, -28], [-20, -10], [-8, 2], [4, 14], [16, 26], [-17, -7]]
    |> dynamic.from
    |> tensor,
  )

  [b2f3d1, negative_biases]
  |> recu(s2d1)
  |> tensor_should_equal(
    [[0, 0], [0, 0], [0, 2], [4, 14], [16, 26], [0, 0]]
    |> dynamic.from
    |> tensor,
  )

  [b2f3d1, negative_biases, b2f3d2, positive_biases]
  |> k_recu(2)(s2d1)
  |> tensor_should_equal(
    [[3, 4], [14, 15], [109, 110], [312, 313], [245, 246], [142, 143]]
    |> dynamic.from
    |> tensor,
  )
}

pub fn cartesian_product_test() {
  let list1 = [500, 1000] |> list.map(dynamic.from)
  let list2 = [0.0001, 0.0002] |> list.map(dynamic.from)
  let list3 = [4, 8] |> list.map(dynamic.from)

  list1
  |> combinatorics.cartesian_product(list2)
  |> list.flat_map(fn(tuple) {
    list3
    |> list.map(fn(item) { #(tuple.0, tuple.1, item) })
  })
  |> should.equal(
    [
      #(500, 0.0001, 4),
      #(500, 0.0001, 8),
      #(500, 0.0002, 4),
      #(500, 0.0002, 8),
      #(1000, 0.0001, 4),
      #(1000, 0.0001, 8),
      #(1000, 0.0002, 4),
      #(1000, 0.0002, 8),
    ]
    |> list.map(fn(item) {
      let #(a, b, c) = item
      #(a |> dynamic.from, b |> dynamic.from, c |> dynamic.from)
    }),
  )

  malt0.cartesian_product([list1, list2, list3])
  |> should.equal(
    [
      #(500, 0.0001, 4),
      #(500, 0.0001, 8),
      #(500, 0.0002, 4),
      #(500, 0.0002, 8),
      #(1000, 0.0001, 4),
      #(1000, 0.0001, 8),
      #(1000, 0.0002, 4),
      #(1000, 0.0002, 8),
    ]
    |> list.map(fn(item) {
      let #(a, b, c) = item
      [a |> dynamic.from, b |> dynamic.from, c |> dynamic.from]
    }),
  )

  // The result will be a list of lists, each containing one element from each input list
  let r =
    malt0.cartesian_product([
      [500, 1000] |> list.map(dynamic.from),
      [0.0001, 0.0002] |> list.map(dynamic.from),
      [4, 8] |> list.map(dynamic.from),
      ["a", "b"] |> list.map(dynamic.from),
    ])

  r |> list.length |> should.equal(16)

  r
  |> iterator.from_list
  |> iterator.at(0)
  |> should.equal(
    [
      500 |> dynamic.from,
      0.0001 |> dynamic.from,
      4 |> dynamic.from,
      "a" |> dynamic.from,
    ]
    |> Ok,
  )

  r
  |> iterator.from_list
  |> iterator.at(15)
  |> should.equal(
    [
      1000 |> dynamic.from,
      0.0002 |> dynamic.from,
      8 |> dynamic.from,
      "b" |> dynamic.from,
    ]
    |> Ok,
  )
}

import iris

pub fn iris_test() {
  iris.print_note()
}
// import qcheck_gleeunit_utils/test_spec
// works, but too SLOW!
// long execution time
// pub fn grid_search_test_() {
//   use <- test_spec.make
//   iris.grid_search_iris_theta()
// }
