"""tests/unit/core 共享测试资产。

FakeClock：注入时钟（M05 spec 任务 6"时间用注入时钟，不真睡"）的最小实现——
读返回 now，advance 推进；限流/熔断的全部时间语义由此可控可断言。
"""


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
