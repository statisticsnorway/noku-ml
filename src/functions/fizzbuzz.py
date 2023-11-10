#!/usr/bin/python3


def fizz(x: int) -> str:
    if x % 3 == 0:
        return "fizz"
    return ""


def buzz(x: int) -> str:
    if x % 5 == 0:
        return "buzz"
    return ""


def fizzbuzz(x: list) -> list:
    result = []
    for y in x:
        r = ""
        r = fizz(y) + buzz(y)
        if not r:
            r = y
        result.append(r)
    return result
