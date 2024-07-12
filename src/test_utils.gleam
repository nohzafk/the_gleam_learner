import gleam/string
import gleeunit/should
import learner.{ListTensor, gradient_operator, is_tensor_equal, tensor_to_list}

pub fn check_theta_and_gradient1(f) {
  fn(t, answers, gradients) {
    should.equal(f(t) |> tensor_to_list, answers |> tensor_to_list)
    should.equal(
      gradient_operator(f, t) |> tensor_to_list,
      gradients |> tensor_to_list,
    )
  }
}

pub fn check_theta_and_gradient2(f) {
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

pub fn check_theta_and_gradient_lossely(f) {
  fn(t, answers, gradients) {
    f(t) |> is_tensor_equal(answers) |> should.be_true

    gradient_operator(f, t) |> is_tensor_equal(gradients) |> should.be_true
  }
}
