//// The Normal Distribution

import gleam/float
import gleam/int
import gleam/list
import gleam_community/maths/elementary.{
  exponential, natural_logarithm, pi, square_root,
}

pub fn random_normal(mu: Float, sigma: Float) -> Float {
  mu +. sigma *. random_standard_normal()
}

pub fn random_standard_normal() -> Float {
  one_normal_random_number(2.0 *. float.random() -. 1.0, int.random(num_parts))
}

//---------------------------------------
// Ziggurat algorithm for converting a
// uniform random number to a normal distribution.
//---------------------------------------
const num_parts = 128

const start_of_tail = 3.442619855899

const area_of_partitions = 9.91256303526217e-3

fn init_boxes(num_boxes: Int, tail: Float, area: Float) -> List(Float) {
  list_of_boxes(
    num_boxes,
    tail,
    area,
    exponential(-0.5 *. start_of_tail *. start_of_tail),
  )
}

fn list_of_boxes(
  num_boxes: Int,
  tail: Float,
  area: Float,
  f: Float,
) -> List(Float) {
  [area /. f, tail, ..more_boxes(num_boxes, area, 2, f, tail, [])]
}

fn more_boxes(
  n: Int,
  area: Float,
  i: Int,
  f: Float,
  last: Float,
  a: List(Float),
) -> List(Float) {
  case i == n {
    True -> [0.0, ..a] |> list.reverse
    False -> {
      let assert Ok(this_sqr) = natural_logarithm(area /. last +. f)
      let assert Ok(this) = square_root(-2.0 *. this_sqr)
      more_boxes(n, area, i + 1, exponential(this_sqr), this, [this, ..a])
    }
  }
}

fn one_normal_random_number(u: Float, i: Int) -> Float {
  case i {
    0 -> normal_tail(start_of_tail, u <. 0.0)
    _ -> non_base_ziggurat_case(u, i)
  }
}

fn non_base_ziggurat_case(u: Float, i: Int) -> Float {
  let boxes = init_boxes(num_parts, start_of_tail, area_of_partitions)

  let assert [b, c, ..] = list.drop(boxes, i)
  let z = u *. b
  case in_the_box(z, c) {
    True -> z
    False ->
      case in_bleed_area(b, c, z) {
        True -> z
        False -> random_standard_normal()
      }
  }
}

fn in_the_box(z: Float, c: Float) -> Bool {
  float.absolute_value(z) <. c
}

fn in_bleed_area(b: Float, c: Float, z: Float) -> Bool {
  let fb = phi(b)
  let fc = phi(c)
  let fz = phi(float.absolute_value(z))
  let box_height = fc -. fb
  let over_height = fz -. fb
  let u1 = float.random()
  u1 *. box_height <. over_height
}

fn normal_tail(min: Float, negative: Bool) -> Float {
  let assert Ok(r1) = natural_logarithm(float.random())
  let x = r1 /. min
  let assert Ok(y) = natural_logarithm(float.random())
  case -2.0 *. y <. x *. x {
    True -> normal_tail(min, negative)
    False ->
      case negative {
        True -> x -. min
        False -> min -. x
      }
  }
}

pub fn phi(x: Float) -> Float {
  let assert Ok(r) = square_root(2.0 *. pi())
  exponential(-0.5 *. x *. x) /. r
}
// this is a programe translated from Racket language to Gleam language
// I've modified a little bit to make it compile without error
// Verify the Gleam implementation carefully about the correctness
