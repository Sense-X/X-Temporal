import abc


class Metric(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    @abc.abstractmethod
    def __ne__(self, other):
        pass

    @abc.abstractmethod
    def __gt__(self, other):
        pass

    @abc.abstractmethod
    def __lt__(self, other):
        pass

    @abc.abstractmethod
    def __ge__(self, other):
        pass

    @abc.abstractmethod
    def __le__(self, other):
        pass


class Top1Metric(Metric):
    def __init__(self, top1, top5, loss=None):
        self.top1 = top1
        self.top5 = top5
        self.loss = loss

    def __str__(self):
        return "Prec@1: %.5f\tPrec@5: %.5f" % (self.top1, self.top5)

    def __repr__(self):
        return "Prec@1: %.5f\tPrec@5: %.5f" % (self.top1, self.top5)

    def __eq__(self, other):
        return self.top1 == other.top1

    def __ne__(self, other):
        return self.top1 != other.top1

    def __gt__(self, other):
        return self.top1 > other.top1

    def __lt__(self, other):
        return self.top1 < other.top1

    def __ge__(self, other):
        return self.top1 >= other.top1

    def __le__(self, other):
        return self.top1 <= other.top1
