import os

notebook_path = os.getcwd()
for folder_level in range(50):
    if "pyproject.toml" in os.listdir():
        break
    os.chdir("../")
from src.functions.fizzbuzz import buzz, fizz, fizzbuzz


def test_fizz():
    assert fizz(3) == "fizz"
    assert fizz(4) == ""


def test_buzz():
    assert buzz(5) == "buzz"
    assert buzz(6) == ""


def test_fizzbuzz():
    assert fizzbuzz([15]) == ["fizzbuzz"]
    assert fizzbuzz([3]) == ["fizz"]
    assert fizzbuzz([5]) == ["buzz"]
    assert fizzbuzz([2]) == [2]
