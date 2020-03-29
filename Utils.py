from typing import Iterable, Callable, Any, TypeVar, Type

T = TypeVar('T')


def find_if(container: Iterable[Type[T]], predicate: Callable[[Any], bool]) -> Type[T]:
    try:
        return next(n for idx, n in enumerate(container) if predicate(n))
    except StopIteration:
        return None
