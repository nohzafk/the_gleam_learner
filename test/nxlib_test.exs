defmodule NxlibTest do
  use ExUnit.Case

  test "add function with debug info" do
    a = {:FloatValue, 3.0}
    b = {:FloatValue, 4.0}

    IO.inspect(a, label: "Input a")
    IO.inspect(b, label: "Input b")

    result = Nxlib.add(a, b)

    IO.inspect(result, label: "Result")

    assert Nx.to_number(result) == 7.0
  end

  test "add function with TensorValue" do
    a = {:TensorValue, Nx.tensor(3.0)}
    b = {:TensorValue, Nx.tensor(4.0)}
    result = Nxlib.add(a, b)
    assert Nx.to_number(result) == 7.0
  end

  test "add function with mixed types" do
    a = {:FloatValue, 3.0}
    b = {:TensorValue, Nx.tensor(4.0)}
    result = Nxlib.add(a, b)
    assert Nx.to_number(result) == 7.0
  end
end
