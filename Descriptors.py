
class BiggerThanZero():
    def __set_name__(self, owner, name) -> None:
        self.name = name

    def __get__(self, instance, owner) -> ...:
        return instance.__dict__.get(self.name, 1)

    def __set__(self, instance, value) -> None:
        if value < 1:
            raise AttributeError('This variable can not be smaller than 1.')
        instance.__dict__[self.name] = value