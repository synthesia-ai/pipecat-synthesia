"""Waiting, frame-filtering, event-recording and secret-sentinel primitives shared by every suite.

Read ``tests/support/README.md`` before writing a test. In particular, never ``await asyncio.sleep(...)``
to wait for something a background task will do; use :func:`tests.support.waiting.until`.
"""
