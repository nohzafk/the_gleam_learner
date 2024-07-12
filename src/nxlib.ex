defmodule Nxlib do
  import Nx.Defn

  def add(a, b) do
    a = to_tensor(a)
    b = to_tensor(b)
    Nx.add(a, b)
  end

  def equal(a, b) do
    a = to_tensor(a)
    b = to_tensor(b)
    Nx.equal(a, b)
  end

  def sum(t) do
    t |> to_tensor |> Nx.sum
  end

  defp to_tensor({:float_value, value}) when is_float(value), do: Nx.tensor(value)
  defp to_tensor({:tensor_value, tensor}), do: tensor
  defp to_tensor(value) when is_float(value), do: Nx.tensor(value)
end
