from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals0 = list(vals)
    vals0[arg] += epsilon

    vals1 = list(vals)
    vals1[arg] -= epsilon

    return (f(*vals0) - f(*vals1)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    Visited = []
    result = []

    def visit(n: Variable):
        if n.is_constant():
            return
        if n.unique_id in Visited:
            return
        if not n.is_leaf():
            for input in n.history.inputs:
                visit(input)
        Visited.append(n.unique_id)
        result.insert(0, n)

    visit(variable)
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    nodes = topological_sort(variable)
    node2deriv = {variable.unique_id: deriv}
    for node in nodes:
        if node.is_leaf():
            continue
        if node.unique_id in node2deriv.keys():
            deriv = node2deriv[node.unique_id]
        deriv_tmp = node.chain_rule(deriv)
        for key, d in deriv_tmp:
            if key.is_leaf():
                key.accumulate_derivative(d)
                continue
            if key.unique_id in node2deriv.keys():
                node2deriv[key.unique_id] += d
            else:
                node2deriv[key.unique_id] = d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
