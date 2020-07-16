class PickBest:
    def __init__(self):
        self.cur = -1
    def update(self, newF1):
        if newF1 > self.cur:
            self.cur = newF1
            return True
        else:
            return False
