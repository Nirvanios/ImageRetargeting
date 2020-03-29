from typing import Tuple, List, Any


class SaliencyObject(object):

    def __init__(self, vertices: List[Tuple]):
        self.triangles = set()
        if not len(vertices) == 3:
            raise AssertionError("Expected 3 points for triangle, got {}.".format(len(vertices)))
        self.triangles.update(vertices)

    def append_if_same(self, to_append: Any) -> bool:
        if type(to_append) is list and all(isinstance(item, tuple) for item in to_append):
            return self.__append_if_same_vertices(to_append)
        elif type(to_append) is SaliencyObject:
            return self.__append_if_same_saliencyobject(to_append)

    def __append_if_same_vertices(self, vertices: List[Tuple]) -> bool:
        if len(self.triangles.intersection(vertices)) >= 2:
            self.triangles.update(vertices)
            return True
        return False

    def __append_if_same_saliencyobject(self, saliency_object: "SaliencyObject") -> bool:
        if len(self.triangles.intersection(saliency_object.triangles)) >= 2:
            self.triangles.update(saliency_object.triangles)
            return True
        return False