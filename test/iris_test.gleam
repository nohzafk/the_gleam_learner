import iris
import qcheck_gleeunit_utils/test_spec

pub fn iris_test() {
  iris.print_note()
}

// long execution time
pub fn grid_search_test_() {
  use <- test_spec.make
  iris.grid_search_iris_theta()
}
