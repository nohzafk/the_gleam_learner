import gleam/dynamic
import gleam/io
import gleam/list
import gleam/string
import iris_data
import nested_tensor.{
  type Block, type Hyperparameters, Block, accuracy, grid_search, init_theta,
  l2_loss, model, relu, sampling_obj, stack_blocks, tensor,
}

//*----------------------------------------
//* Network definitions
//*----------------------------------------
pub fn dense_block(n, m) {
  Block(
    //
    relu,
    [
      //
      [m, n],
      //
      [m],
    ],
  )
}

pub fn iris_network() {
  stack_blocks([
    //
    dense_block(4, 8),
    //
    dense_block(8, 3),
  ])
}

pub fn iris_theta_shapes() {
  let block: Block = iris_network()
  block.block_shape
}

pub fn iris_classifier() {
  let block: Block = iris_network()
  block.block_fn
}

//*----------------------------------------
//* Some warnings
//*----------------------------------------
pub fn print_note() {
  "
--------------------------------------------------------
A WORD OF ADVICE ABOUT IRIS
--------------------------------------------------------
The smallness of iris-network combined with the stochastic nature of init-theta
often causes variations in the final hyperparameters found by grid-search.
This means you might get results that are different from what are shown in the book.
As long as you get a high enough accuracy on the test set, any trained theta is acceptable.
It could also happen that once in a while, no working hyperparameter combination is found.
In that case please run the grid-search again.
Please refer to the Chapter Guide on https://www.thelittlelearner.com for further guidance.
"
  |> string.trim
  |> io.println_error
}

pub fn iris_full_xs() {
  iris_data.iris_full
  |> list.map(fn(data) {
    let assert [xs, _] = data
    xs
  })
  |> dynamic.from
  |> tensor
}

pub fn iris_full_ys() {
  iris_data.iris_full
  |> list.map(fn(data) {
    let assert [_, ys] = data
    ys
  })
  |> dynamic.from
  |> tensor
}

pub fn iris_train_xs() {
  iris_data.iris_train
  |> list.map(fn(data) {
    let assert [xs, _] = data
    xs
  })
  |> dynamic.from
  |> tensor
}

pub fn iris_train_ys() {
  iris_data.iris_train
  |> list.map(fn(data) {
    let assert [_, ys] = data
    ys
  })
  |> dynamic.from
  |> tensor
}

pub fn iris_validate_xs() {
  iris_data.iris_validate
  |> list.map(fn(data) {
    let assert [xs, _] = data
    xs
  })
  |> dynamic.from
  |> tensor
}

pub fn iris_validate_ys() {
  iris_data.iris_validate
  |> list.map(fn(data) {
    let assert [_, ys] = data
    ys
  })
  |> dynamic.from
  |> tensor
}

pub fn iris_test_xs() {
  iris_data.iris_test
  |> list.map(fn(data) {
    let assert [xs, _] = data
    xs
  })
  |> dynamic.from
  |> tensor
}

pub fn iris_test_ys() {
  iris_data.iris_test
  |> list.map(fn(data) {
    let assert [_, ys] = data
    ys
  })
  |> dynamic.from
  |> tensor
}

//*----------------------------------------
//* Grid searching
//*----------------------------------------

pub fn accurate_enough_iris_theta(theta) {
  accuracy(model(iris_classifier(), theta), iris_test_xs(), iris_test_ys())
  >=. 0.9
}

pub fn grid_search_iris_theta(gs) {
  fn(hp: Hyperparameters) {
    { hp |> gs }(
      { hp.batch_size |> sampling_obj }(
        l2_loss(iris_classifier()),
        iris_train_xs(),
        iris_train_ys(),
      ),
      init_theta(iris_theta_shapes()),
    )
  }
  |> grid_search(
    accurate_enough_iris_theta,
    revs: [500, 1000, 2000, 4000],
    alpha: [0.0001, 0.0002, 0.0005],
    batch_size: [4, 8, 16],
  )
}
