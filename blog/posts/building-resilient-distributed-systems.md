---
title: Building Resilient Distributed Systems
date: December 3, 2024
category: Engineering
slug: building-resilient-distributed-systems
description: Patterns and practices for creating services that handle failures gracefully and maintain uptime.
---

# Building Resilient Distributed Systems

In distributed systems, failure isn't an edge case - it's the normal operating condition. Networks partition, services crash, and databases slow down. The question isn't if things will fail, but how your system responds when they do.

Here's what we've learned building systems that stay up when everything around them is falling apart.

## The Fallacies of Distributed Computing

Let's get these out of the way first. These are false assumptions that will bite you:

1. The network is reliable
2. Latency is zero
3. Bandwidth is infinite
4. The network is secure
5. Topology doesn't change
6. There is one administrator
7. Transport cost is zero
8. The network is homogeneous

Every one of these is false. Build accordingly.

## Core Resilience Patterns

### 1. Circuit Breakers

When a service starts failing, stop hammering it. Circuit breakers prevent cascading failures by failing fast.

```python
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
payment_breaker = CircuitBreaker(failure_threshold=5, timeout=30)

def process_payment(amount):
    return payment_breaker.call(payment_service.charge, amount)
```

**Key insight:** Circuit breakers prevent resource exhaustion. When a service is down, you don't waste threads/connections trying to reach it.

### 2. Exponential Backoff with Jitter

When retrying failed requests, don't retry immediately. And don't have all clients retry at the same time.

```python
import random
import time

def exponential_backoff_retry(func, max_retries=5, base_delay=1):
    """
    Retry with exponential backoff and full jitter
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            max_delay = base_delay * (2 ** attempt)
            
            # Full jitter: random delay between 0 and max_delay
            delay = random.uniform(0, max_delay)
            
            time.sleep(delay)

# Usage
result = exponential_backoff_retry(
    lambda: external_api.get_data(),
    max_retries=5,
    base_delay=1
)
```

**Why jitter matters:** Without jitter, all clients retry at the same intervals, creating synchronized thundering herds. Jitter spreads out the load.

### 3. Bulkheads

Isolate resources so failures in one area don't affect others. Like bulkheads in a ship keep it afloat even when one compartment floods.

```python
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore

class BulkheadExecutor:
    """
    Separate thread pools for different services
    """
    def __init__(self):
        # Each service gets its own thread pool
        self.payment_pool = ThreadPoolExecutor(max_workers=10)
        self.email_pool = ThreadPoolExecutor(max_workers=5)
        self.analytics_pool = ThreadPoolExecutor(max_workers=3)
        
        # Semaphores for additional control
        self.payment_semaphore = Semaphore(10)
        self.email_semaphore = Semaphore(5)
    
    def execute_payment(self, func, *args):
        """Payment operations get dedicated resources"""
        with self.payment_semaphore:
            future = self.payment_pool.submit(func, *args)
            return future.result(timeout=30)
    
    def execute_email(self, func, *args):
        """Email can't exhaust payment resources"""
        with self.email_semaphore:
            future = self.email_pool.submit(func, *args)
            return future.result(timeout=10)

# Usage
bulkhead = BulkheadExecutor()

# Even if email service hangs, payments still work
bulkhead.execute_payment(process_payment, amount)
bulkhead.execute_email(send_receipt, email)
```

### 4. Timeouts Everywhere

Every network call needs a timeout. Period.

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_resilient_session():
    """
    Configured session with timeouts and retries
    """
    session = requests.Session()
    
    # Retry logic
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Always use timeouts
session = create_resilient_session()
response = session.get(
    url,
    timeout=(3, 10)  # (connect timeout, read timeout)
)
```

**Common mistake:** No timeout means a slow service can block your threads forever.

### 5. Graceful Degradation

When dependencies fail, provide reduced functionality rather than complete failure.

```python
class UserService:
    def get_user_profile(self, user_id):
        """
        Returns user profile with graceful degradation
        """
        # Critical data from primary DB (required)
        try:
            user = self.db.get_user(user_id)
        except Exception:
            # Can't proceed without basic user data
            raise
        
        # Recommendation engine (nice to have)
        try:
            recommendations = self.recommendation_service.get(user_id)
        except Exception:
            recommendations = []  # Degrade gracefully
        
        # Social stats (optional)
        try:
            social = self.social_service.get_stats(user_id)
        except Exception:
            social = {"followers": 0, "following": 0}  # Default
        
        return {
            "user": user,
            "recommendations": recommendations,
            "social": social
        }
```

## Observability: You Can't Fix What You Can't See

Resilience requires visibility. Implement comprehensive logging, metrics, and tracing.

### Structured Logging

```python
import structlog
import uuid

logger = structlog.get_logger()

def process_order(order_id):
    # Generate correlation ID for tracing
    correlation_id = str(uuid.uuid4())
    
    log = logger.bind(
        correlation_id=correlation_id,
        order_id=order_id
    )
    
    log.info("processing_order_start")
    
    try:
        # Process order...
        result = charge_payment(order_id)
        
        log.info("payment_succeeded", 
                 amount=result.amount,
                 transaction_id=result.id)
        
        return result
        
    except PaymentError as e:
        log.error("payment_failed",
                  error=str(e),
                  error_code=e.code)
        raise
```

### Distributed Tracing

Track requests across services:

```python
from opentelemetry import trace
from opentelemetry.propagate import inject

tracer = trace.get_tracer(__name__)

def call_downstream_service(data):
    with tracer.start_as_current_span("call_payment_service") as span:
        span.set_attribute("payment.amount", data['amount'])
        span.set_attribute("payment.currency", data['currency'])
        
        # Inject trace context into headers
        headers = {}
        inject(headers)
        
        try:
            response = requests.post(
                payment_url,
                json=data,
                headers=headers,
                timeout=5
            )
            span.set_attribute("http.status_code", response.status_code)
            return response
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise
```

## Data Consistency in Distributed Systems

CAP theorem: you can't have Consistency, Availability, and Partition tolerance simultaneously. Pick two.

### Eventual Consistency with Event Sourcing

```python
class OrderEventStore:
    """
    Event sourcing for orders - eventually consistent
    """
    def create_order(self, order_data):
        event = {
            "type": "OrderCreated",
            "order_id": order_data['id'],
            "timestamp": time.time(),
            "data": order_data
        }
        
        # Write to event log (source of truth)
        self.event_log.append(event)
        
        # Publish for eventual propagation
        self.event_bus.publish("orders", event)
        
        return event
    
    def rebuild_order_state(self, order_id):
        """
        Rebuild current state from events
        """
        events = self.event_log.get_events(order_id)
        
        state = {}
        for event in events:
            state = self._apply_event(state, event)
        
        return state
```

### Saga Pattern for Distributed Transactions

```python
class OrderSaga:
    """
    Coordinating distributed transaction across services
    """
    def __init__(self):
        self.steps = [
            (self.reserve_inventory, self.release_inventory),
            (self.charge_payment, self.refund_payment),
            (self.create_shipment, self.cancel_shipment),
        ]
    
    def execute(self, order):
        completed_steps = []
        
        try:
            # Execute forward steps
            for action, compensation in self.steps:
                result = action(order)
                completed_steps.append((compensation, result))
                
            return {"status": "success"}
            
        except Exception as e:
            # Rollback completed steps
            for compensation, result in reversed(completed_steps):
                try:
                    compensation(result)
                except Exception as comp_error:
                    # Log compensation failure
                    logger.error("compensation_failed",
                                error=str(comp_error))
            
            return {"status": "failed", "error": str(e)}
```

## Load Shedding

When overwhelmed, drop less important requests to protect critical functionality.

```python
from enum import IntEnum

class Priority(IntEnum):
    CRITICAL = 1   # Payment processing
    HIGH = 2       # User-facing reads
    MEDIUM = 3     # Analytics
    LOW = 4        # Background jobs

class LoadShedder:
    def __init__(self, max_load=0.8):
        self.max_load = max_load
    
    def should_accept(self, priority: Priority):
        current_load = self.get_current_load()
        
        if current_load < self.max_load:
            return True
        
        # Under load: accept based on priority
        if priority == Priority.CRITICAL:
            return True
        elif priority == Priority.HIGH:
            return current_load < 0.9
        elif priority == Priority.MEDIUM:
            return current_load < 0.95
        else:
            return False  # Drop low priority
    
    def get_current_load(self):
        # CPU + memory + queue depth
        return calculate_system_load()
```

## Testing for Resilience

### Chaos Engineering

Deliberately inject failures to test resilience:

```python
import random

class ChaosMiddleware:
    """
    Randomly inject failures in development/staging
    """
    def __init__(self, failure_rate=0.1):
        self.failure_rate = failure_rate
    
    def __call__(self, request):
        if random.random() < self.failure_rate:
            failure_type = random.choice([
                'timeout',
                'error_500',
                'network_error'
            ])
            
            if failure_type == 'timeout':
                time.sleep(30)  # Simulate timeout
            elif failure_type == 'error_500':
                return Response("Internal Error", status=500)
            else:
                raise ConnectionError("Simulated network failure")
        
        return self.app(request)
```

## Key Principles

1. **Assume failure:** Design for failure as the default state
2. **Fail fast:** Don't waste resources on doomed requests
3. **Isolate failures:** Use bulkheads to contain damage
4. **Degrade gracefully:** Partial functionality > complete failure
5. **Retry smartly:** Exponential backoff with jitter
6. **Timeout everything:** Never wait indefinitely
7. **Monitor constantly:** You can't improve what you don't measure

## Conclusion

Building resilient distributed systems isn't about preventing failures - it's about handling them gracefully. The patterns above aren't optional niceties; they're requirements for production systems.

Start with circuit breakers and timeouts. Add bulkheads for isolation. Implement comprehensive observability. Test with chaos engineering.

Your users won't notice when things work perfectly. They'll definitely notice when they don't.

---

*Running distributed systems in production? What patterns have you found most valuable? Share your experiences in the comments.*
