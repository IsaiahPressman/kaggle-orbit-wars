import numpy as np
from owl.rs import hello_from_rust, hello_numpy


def test_hello_from_rust() -> None:
    assert hello_from_rust() == "Hello from owl!"


def test_hello_numpy() -> None:
    arr = hello_numpy()
    assert arr.shape == (4, 2)
    assert arr.dtype == np.float32
    assert arr[0, 0] == 1
    assert arr[3, 1] == 2
