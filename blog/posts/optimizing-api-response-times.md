---
title: Optimizing API Response Times at Scale
date: December 12, 2024
category: Engineering
slug: optimizing-api-response-times
description: Lessons learned from handling millions of requests per day and the architectural decisions that made it possible.
---

# Optimizing API Response Times at Scale

When your API starts handling millions of requests per day, every millisecond matters. What worked for thousands of requests suddenly becomes a bottleneck. Here's what we learned optimizing our infrastructure to handle serious scale while maintaining sub-100ms response times.

## The Problem

Six months ago, our API response times averaged 350ms. Not terrible, but as traffic grew 10x, we were seeing P95 latencies spike to 2+ seconds during peak hours. Users were complaining, and our infrastructure costs were ballooning.

The root causes weren't immediately obvious. Our code looked fine in isolation. The problem was systemic.

## Key Optimizations

### 1. Database Connection Pooling

**Before:** Creating a new database connection for each request.

**After:** Implemented connection pooling with PgBouncer, maintaining a warm pool of 50 connections.

**Impact:** 200ms → 45ms average database query time.

```python
# Bad: Creating connections per request
def get_user(user_id):
    conn = psycopg2.connect(DATABASE_URL)
    # ... query ...
    conn.close()

# Good: Reusing pooled connections
from psycopg2 import pool
connection_pool = pool.SimpleConnectionPool(10, 50, DATABASE_URL)

def get_user(user_id):
    conn = connection_pool.getconn()
    try:
        # ... query ...
    finally:
        connection_pool.putconn(conn)
```

### 2. Strategic Caching Layers

We implemented a three-tier caching strategy:

- **L1: In-memory cache** (Redis) for hot data (< 1ms access)
- **L2: CDN edge cache** for static and semi-static responses
- **L3: Application-level cache** for computed results

The key insight: don't cache everything. Cache what's expensive to compute and frequently accessed.

```python
import redis
from functools import wraps

cache = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(ttl=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # Try cache first
            cached = cache.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

@cache_result(ttl=600)
def get_user_stats(user_id):
    # Expensive aggregation query
    return compute_stats(user_id)
```

### 3. Async I/O Where It Matters

For endpoints making multiple external API calls, we switched from sequential to parallel requests using async/await.

**Sequential (slow):**
```python
def get_dashboard_data(user_id):
    user = fetch_user(user_id)        # 50ms
    orders = fetch_orders(user_id)    # 80ms
    stats = fetch_stats(user_id)      # 120ms
    return combine(user, orders, stats)
    # Total: 250ms
```

**Parallel (fast):**
```python
async def get_dashboard_data(user_id):
    user, orders, stats = await asyncio.gather(
        fetch_user(user_id),
        fetch_orders(user_id),
        fetch_stats(user_id)
    )
    return combine(user, orders, stats)
    # Total: 120ms (longest single request)
```

### 4. Database Query Optimization

We profiled every query using `EXPLAIN ANALYZE` and found several issues:

- **Missing indexes** on frequently filtered columns
- **N+1 queries** in ORM code
- **Unnecessary JOINs** that could be split into separate cached queries

One particularly bad endpoint was doing 47 database queries. We reduced it to 3 with strategic eager loading and denormalization.

### 5. Smart Rate Limiting

Rather than rejecting requests outright, we implemented adaptive rate limiting that degrades gracefully:

```python
class AdaptiveRateLimiter:
    def __init__(self):
        self.redis = redis.Redis()
        
    def check_limit(self, user_id, endpoint):
        key = f"rate:{user_id}:{endpoint}"
        
        # Get current system load
        load = self.get_system_load()
        
        # Adjust limits based on load
        if load > 0.8:
            limit = 100  # Strict during high load
        else:
            limit = 500  # Generous during low load
            
        current = self.redis.incr(key)
        if current == 1:
            self.redis.expire(key, 60)
            
        return current <= limit
```

## Results

After implementing these optimizations:

- **Average response time:** 350ms → 78ms
- **P95 latency:** 2000ms → 180ms
- **Throughput:** 3x increase with same infrastructure
- **Cost:** 40% reduction in database costs

## Key Takeaways

1. **Measure everything:** We added detailed timing middleware to identify bottlenecks. You can't optimize what you don't measure.

2. **Cache strategically:** Don't cache everything. Cache expensive, frequently-accessed data with appropriate TTLs.

3. **Connection pooling is mandatory:** At scale, connection overhead dominates. Pool everything - databases, HTTP clients, external services.

4. **Async doesn't solve everything:** Only use async for I/O-bound operations. For CPU-bound work, it can actually make things slower.

5. **Database optimization > code optimization:** In most cases, a missing index has 100x more impact than optimizing your Python code.

## What's Next

We're now working on:
- **Geographic distribution:** Multi-region deployment to reduce latency globally
- **Query result streaming:** For large datasets, stream results rather than buffering
- **Predictive caching:** Using ML to pre-cache data users are likely to request

Performance optimization is never "done" - it's an ongoing process of measurement, hypothesis, and iteration. But these fundamentals will take you far.

---

*Have questions about API optimization? Found this helpful? Let me know in the comments or reach out on Twitter.*
