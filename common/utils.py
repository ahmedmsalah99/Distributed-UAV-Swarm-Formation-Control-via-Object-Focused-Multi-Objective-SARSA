from dataclasses import dataclass
import math

@dataclass
class Point:
    x: float
    y: float

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        raise TypeError("Can only add Point to Point")

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        raise TypeError("Can only subtract Point from Point")
    def __mul__(self,num):
        return Point(self.x * num, self.y * num)
    def dist(self,other):
        return math.sqrt((self.x-other.x)**2+(self.y-other.y)**2)
    def angle(self,other):
        vec_x = other.x - self.x
        vec_y = other.y - self.y
        return math.atan2(vec_y/vec_x)* 180/math.pi
    def to_array(self):
        return [self.x,self.y]


