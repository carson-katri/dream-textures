def block_in_use(func):
    def block(self, *args, **kwargs):
        if self.in_use:
            raise RuntimeError(f"Can't call {func.__qualname__} while process is in use")
        try:
            self.in_use = True
            yield from func(self, *args, **kwargs)
        finally:
            self.in_use = False
    # Pass the name through so we can use it in `setattr` on `GeneratorProcess`.
    block.__name__ = func.__name__
    return block