__all__ = ['Module']

class Module(object):
    def __init__(self, savename=None, per_order=False, **kwargs):
        self.savename = savename
        self.per_order = per_order
        self.per_order_possible = True
        self.initialise(**kwargs)

    def initialise(self):
        pass

    def __call__(self, dataset, debug=False):
        return self.process(dataset, debug=debug)

    def process(self, dataset, debug=False):
        return dataset
