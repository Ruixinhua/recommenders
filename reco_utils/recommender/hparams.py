class HParams:

    def __init__(self, **kwargs):
        self.params = sorted(kwargs.items(), key=lambda i: i[0])
        self.__dict__.update(kwargs)

    def to_string(self):
        return ",".join([f"({k}, {v})" for k, v in self.params])

    def update(self, **kwargs):
        self.params = sorted(kwargs.items(), key=lambda i: i[0])
        self.__dict__.update(kwargs)
        return self
