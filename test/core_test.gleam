import core.{
  adam_gradient_descent, cross_entropy_loss, gradient_descent, hp_new,
  hp_new_batch_size, hp_new_beta, hp_new_mu, kl_loss, l2_loss, line, linear,
  naked_gradient_descent, plane, quad, revise, rms_gradient_descent, samples,
  sampling_obj, smooth, tensor_dot_product, tensor_dot_product_2_1,
  tensor_to_float, velocity_gradient_descent, zeros,
}
import gleam/dynamic
import gleam/float
import gleam/function
import gleam/list
import gleeunit/should
import learner.{
  type Tensor, ListTensor, ScalarTensor, gradient_of, tensor, tensor_add,
  tensor_divide, tensor_exp, tensor_log, tensor_minus, tensor_multiply,
  tensor_multiply_2_1, tensor_sqr, tensor_sqrt, tensor_sum, tensor_to_list,
  tensorized_cmp_equal, to_tensor,
}
import test_utils.{check_theta_and_gradient1, check_theta_and_gradient_lossely}

pub fn lift_to_tensor1(f: fn(Tensor) -> Tensor) -> fn(Float) -> Float {
  fn(v) {
    let assert ScalarTensor(s) = v |> to_tensor |> f
    s.real
  }
}

pub fn lift_to_tensor2(
  f: fn(Tensor, Tensor) -> Tensor,
) -> fn(Float, Float) -> Float {
  fn(a, b) {
    let assert ScalarTensor(s) = f(a |> to_tensor, b |> to_tensor)
    s.real
  }
}

fn tensor_op_wrapper(f) {
  fn(t) {
    let assert ListTensor([a, b]) = t
    f(a, b)
  }
}

fn tensor_should_equal(actual, expected) {
  should.equal(actual |> tensor_to_list, expected |> tensor_to_list)
}

pub fn a_core_test() {
  {
    let a = 7.0
    let b = 13.0
    let t = [a, b] |> dynamic.from |> tensor

    { tensor_add |> lift_to_tensor2 }(a, b)
    |> should.equal(20.0)

    { tensor_add |> tensor_op_wrapper }
    |> gradient_of(t)
    |> tensor_should_equal([1.0, 1.0] |> dynamic.from |> tensor)

    { tensor_multiply |> lift_to_tensor2 }(a, b)
    |> should.equal(91.0)

    { tensor_multiply |> tensor_op_wrapper }
    |> gradient_of(t)
    |> tensor_should_equal([13.0, 7.0] |> dynamic.from |> tensor)

    { tensor_divide |> tensor_op_wrapper }
    |> gradient_of(t)
    |> learner.is_tensor_equal([0.07692, -0.04142] |> dynamic.from |> tensor)
    |> should.be_true
  }

  {
    let a = [7, 8, 9] |> dynamic.from |> tensor

    tensor_exp(a)
    |> learner.is_tensor_equal(
      [1096.6331, 2980.9579, 8103.0839] |> dynamic.from |> tensor,
    )
    |> should.be_true

    tensor_exp
    |> gradient_of(a)
    |> learner.is_tensor_equal(
      [1096.6331, 2980.9579, 8103.0839] |> dynamic.from |> tensor,
    )
    |> should.be_true

    tensor_log(a)
    |> learner.is_tensor_equal(
      [1.9459, 2.0794, 2.1972] |> dynamic.from |> tensor,
    )
    |> should.be_true

    tensor_log
    |> gradient_of(a)
    |> learner.is_tensor_equal(
      [0.1428, 0.125, 0.1111] |> dynamic.from |> tensor,
    )
    |> should.be_true

    tensor_sqrt(a)
    |> learner.is_tensor_equal([2.6457, 2.8284, 3.0] |> dynamic.from |> tensor)
    |> should.be_true
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

pub fn make_theta(weights: dynamic.Dynamic, bias: Float) {
  ListTensor([weights |> tensor, bias |> to_tensor])
}

pub fn b_targets_test() {
  let r1d1 = [3, 4, 5] |> dynamic.from |> tensor
  let r1d2 = [7, 8, 9] |> dynamic.from |> tensor
  let r2d1 =
    [[3, 4, 5], [3, 4, 5]]
    |> dynamic.from
    |> tensor
  let r2d2 =
    [[7, 8, 9], [7, 8, 9]]
    |> dynamic.from
    |> tensor

  let line_theta_1 = [1.0, 2.0] |> dynamic.from |> tensor

  { line(r1d1) |> check_theta_and_gradient1 }(
    line_theta_1,
    [5, 6, 7] |> dynamic.from |> tensor,
    [12, 3] |> dynamic.from |> tensor,
  )

  { line(r1d2) |> check_theta_and_gradient1 }(
    line_theta_1,
    [9, 10, 11] |> dynamic.from |> tensor,
    [24, 3] |> dynamic.from |> tensor,
  )

  { line(r2d1) |> check_theta_and_gradient1 }(
    line_theta_1,
    [[5, 6, 7], [5, 6, 7]]
      |> dynamic.from
      |> tensor,
    [24, 6] |> dynamic.from |> tensor,
  )

  let quad_theta_1 = [14, 2, 8] |> dynamic.from |> tensor

  { quad(r1d1) |> check_theta_and_gradient1 }(
    quad_theta_1,
    [140, 240, 368] |> dynamic.from |> tensor,
    [50, 12, 3] |> dynamic.from |> tensor,
  )

  { quad(r1d2) |> check_theta_and_gradient1 }(
    quad_theta_1,
    [708.0, 920.0, 1160.0] |> dynamic.from |> tensor,
    [194, 24, 3] |> dynamic.from |> tensor,
  )

  { quad(r2d1) |> check_theta_and_gradient1 }(
    quad_theta_1,
    [[140, 240, 368], [140, 240, 368]]
      |> dynamic.from
      |> tensor,
    [100, 24, 6] |> dynamic.from |> tensor,
  )

  { quad(r2d2) |> check_theta_and_gradient1 }(
    quad_theta_1,
    [[708.0, 920.0, 1160.0], [708.0, 920.0, 1160.0]]
      |> dynamic.from
      |> tensor,
    [388, 48, 6] |> dynamic.from |> tensor,
  )

  let plane_theta_1 = make_theta([9, 8, 7] |> dynamic.from, 2.0)

  { plane(r1d1) |> check_theta_and_gradient1 }(
    plane_theta_1,
    to_tensor(96.0),
    make_theta([3, 4, 5] |> dynamic.from, 1.0),
  )

  { plane(r2d1) |> check_theta_and_gradient1 }(
    plane_theta_1,
    [96, 96] |> dynamic.from |> tensor,
    make_theta([6, 8, 10] |> dynamic.from, 2.0),
  )

  linear([-1, 0, 1] |> dynamic.from |> tensor)(make_theta(
    [[1, 2, 3], [1, 2, 3]]
      |> dynamic.from,
    2.0,
  ))
  |> tensor_should_equal([4.0, 4.0] |> dynamic.from |> tensor)

  linear(
    [[-1, 0, 1], [-1, 0, 1]]
    |> dynamic.from
    |> tensor,
  )(make_theta(
    [[1, 2, 3], [1, 2, 3]]
      |> dynamic.from,
    2.0,
  ))
  |> tensor_should_equal(
    [[4, 4], [4, 4]]
    |> dynamic.from
    |> tensor,
  )

  linear(
    [[-1, 0, 1], [-1, 0, 2]]
    |> dynamic.from
    |> tensor,
  )
  |> gradient_of(make_theta(
    [[1, 2, 3], [1, 2, 4]]
      |> dynamic.from,
    2.0,
  ))
  |> tensor_should_equal(make_theta(
    [[-2, 0, 3], [-2, 0, 3]]
      |> dynamic.from,
    4.0,
  ))

  linear(
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    |> dynamic.from
    |> tensor,
  )(make_theta(
    [[1, 2, 3], [1, 2, 3]]
      |> dynamic.from,
    2.0,
  ))
  |> tensor_should_equal(
    [[4, 4], [4, 4], [4, 4]]
    |> dynamic.from
    |> tensor,
  )

  linear(
    [[-1, 0, 1], [-1, 0, 1]]
    |> dynamic.from
    |> tensor,
  )
  |> gradient_of(make_theta(
    [[1, 2, 3], [1, 2, 3]]
      |> dynamic.from,
    2.0,
  ))
  |> tensor_should_equal(make_theta(
    [[-2, 0, 2], [-2, 0, 2]]
      |> dynamic.from,
    4.0,
  ))
}

pub fn c_loss_test() {
  let r2d1 =
    [[3, 4, 5], [3, 4, 5]]
    |> dynamic.from
    |> tensor

  let plane_theta_0 = make_theta([0, 0, 0] |> dynamic.from, 0.0)

  {
    l2_loss(plane)(r2d1, [1.0, 1.0] |> dynamic.from |> tensor)
    |> check_theta_and_gradient1
  }(
    plane_theta_0,
    to_tensor(2.0),
    make_theta(
      [-12, -16, -20]
        |> dynamic.from,
      -4.0,
    ),
  )

  let dist1 =
    [[0.3, 0.4, 0.3], [0.1, 0.1, 0.8]]
    |> dynamic.from
    |> tensor

  let ce_theta =
    [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]]
    |> dynamic.from
    |> tensor

  let bad = fn(t) {
    fn(theta) {
      let assert ListTensor([a, b]) = theta
      tensor_multiply(a, t) |> tensor_add(b)
    }
  }

  { cross_entropy_loss(bad)(dist1, dist1) |> check_theta_and_gradient_lossely }(
    ce_theta,
    [0.49221825504118605, 0.6033077198687743] |> dynamic.from |> tensor,
    [
      [-0.03178270152963002, -0.47619047619047616, -1.7846790890269149],
      [-0.1309111274458329, -1.4285714285714284, -2.774327122153209],
    ]
      |> dynamic.from
      |> tensor,
  )

  { kl_loss(bad)(dist1, dist1) |> check_theta_and_gradient_lossely }(
    ce_theta,
    [1.1059011281529316, 1.706692900826487] |> dynamic.from |> tensor,
    [
      [1.0000945635137346, 0.023289894686568724, -0.5820305479347834],
      [5.457682729537845, 0.8448173598434958, -0.701819651351574],
    ]
      |> dynamic.from
      |> tensor,
  )
}

pub fn gradient_descent_test() {
  revise(fn(x) { x + 1 }, 10, 0) |> should.equal(10)

  let obj = fn(theta) {
    let assert ListTensor([a, ..]) = theta
    to_tensor(30.0) |> tensor_minus(a) |> tensor_sqr
  }

  let id = function.identity

  let hp = hp_new(revs: 500, alpha: 0.01)

  let naked_gd =
    gradient_descent(hp, id, id, fn(hp) {
      fn(w, g) {
        tensor_multiply(g, hp.alpha |> to_tensor)
        |> tensor_minus(w, _)
      }
    })

  naked_gd(obj, [3.0] |> dynamic.from |> tensor)
  |> learner.is_tensor_equal([29.998892352401082] |> dynamic.from |> tensor)
}

pub fn d_gd_common_test() {
  zeros([1, 2, 3] |> dynamic.from |> tensor)
  |> tensor_should_equal([0, 0, 0] |> dynamic.from |> tensor)

  smooth(0.9, 31.0, -8.0) |> should.equal(27.1)
}

pub fn f_naked_test() {
  let obj = fn(theta) {
    let assert ListTensor([a]) = theta
    tensor_minus(30.0 |> to_tensor, a) |> tensor_sqr
  }

  let hp = hp_new(revs: 400, alpha: 0.01)
  { hp |> naked_gradient_descent }(obj, [3.0] |> dynamic.from |> tensor)
  |> learner.is_tensor_equal([29.991647931623252] |> dynamic.from |> tensor)
}

pub fn g_velocity_gradient_descent() {
  let obj = fn(theta) {
    let assert ListTensor([a]) = theta
    tensor_minus(30.0 |> to_tensor, a) |> tensor_sqr
  }

  let hp = hp_new(revs: 70, alpha: 0.01) |> hp_new_mu(mu: 0.9)
  { hp |> velocity_gradient_descent }(obj, [3.0] |> dynamic.from |> tensor)
  |> learner.is_tensor_equal([30.686162582787535] |> dynamic.from |> tensor)
}

pub fn h_rms_gradient_descent_test() {
  let obj = fn(theta) {
    let assert ListTensor([a]) = theta
    tensor_minus(30.0 |> to_tensor, a) |> tensor_sqr
  }

  let hp = hp_new(revs: 170, alpha: 0.1) |> hp_new_beta(beta: 0.999)
  { hp |> rms_gradient_descent }(obj, [3.0] |> dynamic.from |> tensor)
  |> learner.is_tensor_equal([29.990436454407956] |> dynamic.from |> tensor)
}

pub fn i_adam_test() {
  let obj = fn(theta) {
    let assert ListTensor([ScalarTensor(a)]) = theta
    let assert Ok(r) = { 30.0 -. a.real } |> float.power(2.0)
    r |> to_tensor
  }
  let hp =
    hp_new(revs: 150, alpha: 0.1)
    |> hp_new_beta(beta: 0.999)
    |> hp_new_mu(mu: 0.9)
  { hp |> adam_gradient_descent }(obj, [3.0] |> dynamic.from |> tensor)
  |> learner.is_tensor_equal([29.994907156105718] |> dynamic.from |> tensor)
}

pub fn j_stochastic_test() {
  let test_samples = samples(10, 3)

  test_samples |> list.length |> should.equal(3)

  test_samples
  |> list.all(fn(x) { x >= 0 && x <= 9 })
  |> should.be_true

  let test_expectant_fn = fn(xs, ys) {
    fn(theta) {
      should.equal(learner.shape(xs), learner.shape(ys))

      tensorized_cmp_equal(xs, ys)
      |> tensor_sum
      |> tensor_to_float
      |> should.equal(3.0)

      theta
    }
  }

  let test_tensor = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] |> dynamic.from |> tensor

  let hp = hp_new(1, 0.1) |> hp_new_batch_size(batch_size: 3)
  { hp |> sampling_obj }(test_expectant_fn, test_tensor, test_tensor)(False)
}
