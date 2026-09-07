"""精确前缀的教学成本模型：单位不是供应商 token，价格不是报价。"""

from dataclasses import dataclass
import math
from .types import units


@dataclass
class Prices:
    write: float = 1.25
    read: float = 0.1

    def __post_init__(self):
        if not all(math.isfinite(x) and x >= 0 for x in (self.write, self.read)):
            raise ValueError("prices must be finite and nonnegative")


class PrefixCache:
    def __init__(self, ttl=300, prices=None):
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("TTL must be finite and positive")
        self.ttl, self.prices, self.entries = ttl, prices or Prices(), {}

    def request(self, text, *, model="teaching", now=0):
        tokens = units(text)
        best = 0
        self.entries = {k: expiry for k, expiry in self.entries.items() if expiry > now}
        for (scope, previous), expiry in self.entries.items():
            if scope != model or expiry <= now:
                continue
            common = 0
            for a, b in zip(tokens, previous):
                if a != b:
                    break
                common += 1
            best = max(best, common)
        self.entries[(model, tuple(tokens))] = now + self.ttl
        return {
            "units": len(tokens),
            "read": best,
            "write": len(tokens) - best,
            "cost": best * self.prices.read + (len(tokens) - best) * self.prices.write,
        }


def commit_value(old_suffix, new_suffix, *, future_calls=1, expired=False, remaining_cost=0, prices=None):
    """比较首个改变位置之后的后缀。已支付 planner 成本不应再次阻止提交。"""
    if (
        not all(math.isfinite(x) for x in (future_calls, old_suffix, new_suffix, remaining_cost))
        or future_calls < 1
        or min(old_suffix, new_suffix, remaining_cost) < 0
    ):
        raise ValueError("invalid cost inputs")
    p = prices or Prices()
    saved = old_suffix - new_suffix
    benefit = (
        saved * p.write + (future_calls - 1) * saved * p.read
        if expired
        else future_calls * saved * p.read - new_suffix * (p.write - p.read)
    )
    return benefit - remaining_cost


class ZonedCache:
    """Exact checkpoint simulator with per-zone TTL. Later checkpoints include all prior bytes."""

    def __init__(self, prices=None, uncached_price=1.0):
        self.prices, self.uncached_price = prices or Prices(), uncached_price
        self.entries = {}

    def request(self, zones, *, model="teaching", now=0):
        # zones = [(name, serialized_text, ttl_seconds_or_0)]
        prefix, endpoints = "", []
        for name, text, ttl in zones:
            if ttl < 0:
                raise ValueError("negative TTL")
            prefix += text
            endpoints.append((name, prefix, ttl))
        self.entries = {k: v for k, v in self.entries.items() if v > now}
        hits = [
            len(units(text))
            for (scope, text), expiry in self.entries.items()
            if scope == model and prefix.startswith(text) and expiry > now
        ]
        read = max(hits, default=0)
        cacheable = max((len(units(text)) for _, text, ttl in endpoints if ttl), default=0)
        # A matched prior checkpoint can extend beyond today's requested write checkpoints.
        read = min(read, len(units(prefix)))
        write = max(0, cacheable - read)
        uncached = max(0, len(units(prefix)) - max(read, cacheable))
        for _, text, ttl in endpoints:
            if ttl:
                self.entries[(model, text)] = now + ttl
        return {
            "read": read,
            "write": write,
            "uncached": uncached,
            "units": len(units(prefix)),
            "cost": read * self.prices.read + write * self.prices.write + uncached * self.uncached_price,
            "checkpoints": [{"zone": n, "end": len(units(t)), "ttl": ttl} for n, t, ttl in endpoints],
        }


def forecast_value(
    old_suffix,
    new_suffix,
    *,
    hit_probability=1.0,
    future_calls=1,
    planner_cost=0.0,
    remaining_cost=0.0,
    prices=None,
):
    """Planning-time expected value. Planner cost is included here, not paid again at commit."""
    if not 0 <= hit_probability <= 1 or planner_cost < 0:
        raise ValueError("invalid forecast parameters")
    valid = commit_value(
        old_suffix, new_suffix, future_calls=future_calls, remaining_cost=remaining_cost, prices=prices
    )
    expired = commit_value(
        old_suffix,
        new_suffix,
        future_calls=future_calls,
        expired=True,
        remaining_cost=remaining_cost,
        prices=prices,
    )
    return hit_probability * valid + (1 - hit_probability) * expired - planner_cost
