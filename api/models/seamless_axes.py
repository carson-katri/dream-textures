from enum import Enum

class SeamlessAxes(Enum):
    """Unified handling of seamless axes.
    Can be converted from str (id or text) or bool tuple/list (x, y).
    Each enum is equal to their respective convertible values.
    Special cases:
        AUTO: None
        OFF: False, empty str
        BOTH: True
    """

    AUTO =       'auto', 'Auto-detect', None,  None
    OFF =        'off',  'Off',         False, False
    HORIZONTAL = 'x',    'X',           True,  False
    VERTICAL =   'y',    'Y',           False, True
    BOTH =       'xy',   'Both',        True,  True

    def __init__(self, id, text, x, y):
        self.id = id
        self.text = text
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self is other
        if isinstance(other, str):
            return self.id == other or self.text == other or (other == '' and self is self.OFF)
        if isinstance(other, (tuple, list)) and len(other) == 2:
            return self.x == other[0] and self.y == other[1]
        if other is True and self is self.BOTH:
            return True
        if other is False and self is self.OFF:
            return True
        if other is None and self is self.AUTO:
            return True
        return False

    def __and__(self, other):
        return SeamlessAxes((self.x and other.x, self.y and other.y))

    def __or__(self, other):
        return SeamlessAxes((self.x or other.x, self.y or other.y))

    def __xor__(self, other):
        return SeamlessAxes((self.x != other.x, self.y != other.y))

    def __invert__(self):
        return SeamlessAxes((not self.x, not self.y))

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            if value == '':
                return cls.OFF
            for e in cls:
                if e.id == value or e.text == value:
                    return e
            raise ValueError(f'no {cls.__name__} with id {repr(id)}')
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            for e in cls:
                if e.x == value[0] and e.y == value[1]:
                    return e
            raise ValueError(f'no {cls.__name__} with x {value[0]} and y {value[1]}')
        elif value is True:
            return cls.BOTH
        elif value is False:
            return cls.OFF
        elif value is None:
            return cls.AUTO
        raise TypeError(f'expected str, bool, tuple[bool, bool], or None, got {repr(value)}')

    def bpy_enum(self, *args):
        return self.id, self.text, *args