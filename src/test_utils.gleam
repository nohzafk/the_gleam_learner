import gleeunit/should
import gleam/string
import learner.{ListTensor, gradient_operator, tensor_to_list}

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
