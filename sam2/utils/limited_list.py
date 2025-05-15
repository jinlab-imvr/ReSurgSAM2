class QueueList(list):
    def __init__(self, maxlen):
        super().__init__()
        self.maxlen = maxlen

    def append(self, item):
        if len(self) >= self.maxlen:
            self.pop(0)  # 移除第一个元素
        super().append(item)  # 插入到最后面

    def appendleft(self, item):
        if len(self) >= self.maxlen:
            self.pop(-1)  # 移除最后一个元素
        super().insert(0, item)  # 插入到最前面

    def pop(self, index=-1):
        if len(self) == 0:
            return
        return super().pop(index)

    # 清空列表
    def clear(self):
        self.__init__(self.maxlen)

