"""
高可用模块.

提供限流器、熔断器、健康检查等高可用组件
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class RateLimitConfig:
    """限流配置."""

    requests_per_second: float = 10.0
    burst_size: int = 20


class RateLimiter:
    """令牌桶限流器.

    Args:
        config: 限流配置
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._tokens = self.config.burst_size
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        """补充令牌."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self.config.requests_per_second
        self._tokens = min(self.config.burst_size, self._tokens + new_tokens)
        self._last_refill = now

    def acquire(self) -> bool:
        """尝试获取令牌.

        Returns:
            是否获取成功
        """
        with self._lock:
            self._refill()
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False

    @property
    def available_tokens(self) -> float:
        """当前可用令牌数."""
        with self._lock:
            self._refill()
            return self._tokens


@dataclass
class CircuitBreakerConfig:
    """熔断器配置."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_requests: int = 3


class CircuitBreaker:
    """熔断器.

    状态机：closed -> open -> half-open -> closed/open

    Args:
        config: 熔断器配置
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._state = "closed"
        self._failure_count = 0
        self._last_failure_time = 0
        self._half_open_requests = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """当前状态."""
        with self._lock:
            if self._state == "open":
                # 检查是否可以转为 half-open
                if time.monotonic() - self._last_failure_time >= self.config.recovery_timeout:
                    self._state = "half-open"
                    self._half_open_requests = 0
            return self._state

    def call(self, func: Callable, *args, **kwargs):
        """通过熔断器调用函数.

        Args:
            func: 要调用的函数
            *args, **kwargs: 函数参数

        Returns:
            函数返回值

        Raises:
            RuntimeError: 熔断器打开时
            Exception: 函数执行异常时
        """
        current_state = self.state

        if current_state == "open":
            raise RuntimeError("熔断器已打开，请求被拒绝")

        if current_state == "half-open":
            with self._lock:
                if self._half_open_requests >= self.config.half_open_max_requests:
                    raise RuntimeError("熔断器半开状态，请求已满")
                self._half_open_requests += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self):
        """调用成功."""
        with self._lock:
            if self._state == "half-open":
                self._state = "closed"
            self._failure_count = 0

    def _on_failure(self):
        """调用失败."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.config.failure_threshold:
                self._state = "open"

    def reset(self):
        """重置熔断器."""
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._half_open_requests = 0


class HealthChecker:
    """健康检查器.

    注册多个检查项，统一检查健康状态
    """

    def __init__(self):
        self._checks: dict[str, Callable[[], bool]] = {}

    def register(self, name: str, check_func: Callable[[], bool]):
        """注册健康检查项.

        Args:
            name: 检查项名称
            check_func: 检查函数，返回 True 表示健康
        """
        self._checks[name] = check_func

    def check(self) -> dict:
        """执行所有健康检查.

        Returns:
            健康检查结果，包含 status 和 details
        """
        details = {}
        all_healthy = True

        for name, check_func in self._checks.items():
            try:
                healthy = check_func()
                details[name] = {"healthy": healthy}
                if not healthy:
                    all_healthy = False
            except Exception as e:
                details[name] = {"healthy": False, "error": str(e)}
                all_healthy = False

        return {"status": "healthy" if all_healthy else "degraded", "details": details}


if __name__ == "__main__":
    # 测试代码
    print("高可用模块测试")

    # 限流器测试
    config = RateLimitConfig(requests_per_second=5)
    limiter = RateLimiter(config)
    success_count = sum(1 for _ in range(10) if limiter.acquire())
    print(f"限流器: 10次请求成功 {success_count} 次")

    # 熔断器测试
    cb_config = CircuitBreakerConfig(failure_threshold=3)
    breaker = CircuitBreaker(cb_config)
    for _ in range(3):
        try:
            breaker.call(lambda: 1 / 0)
        except Exception:
            pass
    print(f"熔断器状态: {breaker.state}")

    # 健康检查测试
    checker = HealthChecker()
    checker.register("test", lambda: True)
    result = checker.check()
    print(f"健康检查: {result['status']}")
