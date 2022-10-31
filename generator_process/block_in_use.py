def block_in_use(func):
    def block(self, *args, **kwargs):
        if self.in_use:
            raise RuntimeError(f"Can't call {func.__qualname__} while process is in use")
        self.in_use = True

        # generator function is separate so in_use gets set immediately rather than waiting for first next() call
        def sub():
            try:
                yield from func(self, *args, **kwargs)
            finally:
                self.in_use = False
        return sub()

    # Pass the name through so we can use it in `setattr` on `GeneratorProcess`.
    block.__name__ = func.__name__
    return block