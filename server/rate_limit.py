"""进程内滑动窗口限流（单机演示用；多实例需换 Redis，见设计文档 §12）。"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from app import config


class SlidingWindowLimiter:
    def __init__(self, per_minute: int):
        self.limit = per_minute
        self.buckets: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> tuple[bool, int]:
        """返回 (是否放行, 剩余额度)。limit<=0 表示关闭。"""
        if self.limit <= 0:
            return True, self.limit
        now = time.monotonic()
        q = self.buckets[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.limit:
            return False, 0
        q.append(now)
        return True, self.limit - len(q)


limiter = SlidingWindowLimiter(config.RATE_LIMIT_PER_MINUTE)
