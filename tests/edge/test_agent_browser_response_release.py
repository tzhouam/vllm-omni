"""Navigation-response cleanup with synchronous API doubles; no browser starts.

These tests exercise the actual ManagedEdgeBrowser methods after application of
the accompanying candidate. They do not prove browser RSS, physical RAM bounds,
real Playwright execution or the cause of any recorded admission refusal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.edge.agent.tools import ManagedEdgeBrowser


class Response:
    def __init__(
        self,
        events: list[str],
        *,
        redirect: bool = False,
        metadata_failure: str | None = None,
        primary_error: BaseException | None = None,
        cleanup_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.redirect = redirect
        self.metadata_failure = metadata_failure
        self.primary_error = primary_error
        self.cleanup_error = cleanup_error
        self.dispose_attempts = 0
        self.disposed = False

    @property
    def headers(self) -> dict[str, str]:
        self.events.append("headers")
        if self.metadata_failure == "headers":
            assert self.primary_error is not None
            raise self.primary_error
        return {"location": "/other"} if self.redirect else {}

    @property
    def status(self) -> int:
        self.events.append("status")
        if self.metadata_failure == "status":
            assert self.primary_error is not None
            raise self.primary_error
        return 302 if self.redirect else 200

    def dispose(self) -> None:
        self.events.append("dispose")
        self.dispose_attempts += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error
        self.disposed = True


class Route:
    def __init__(
        self,
        browser: ManagedEdgeBrowser,
        response: Response,
        *,
        fetch_error: BaseException | None = None,
        fulfill_error: BaseException | None = None,
        abort_error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.events = response.events
        self.fetch_error = fetch_error
        self.fulfill_error = fulfill_error
        self.abort_error = abort_error
        self.request = SimpleNamespace(
            is_navigation_request=lambda: True,
            frame=browser._page.main_frame,
            method="GET",
            url="https://example.test/report",
        )

    def fetch(self, **kwargs: object) -> Response:
        assert kwargs == {"max_redirects": 0, "timeout": 30_000}
        self.events.append("fetch")
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.response

    def fulfill(self, *, response: Response) -> None:
        assert response is self.response
        assert response.dispose_attempts == 0
        assert not response.disposed
        self.events.append("fulfill_consumes_body")
        if self.fulfill_error is not None:
            raise self.fulfill_error
        self.events.append("fulfill_returned")

    def abort(self, reason: str) -> None:
        self.events.append("abort:" + reason)
        if self.abort_error is not None:
            raise self.abort_error


def browser() -> ManagedEdgeBrowser:
    # Avoid the Windows/browser constructor; all tested methods are real.
    result = object.__new__(ManagedEdgeBrowser)
    result._page = SimpleNamespace(main_frame=object())
    result._navigation_target = "https://example.test/report"
    result._navigation_target_used = False
    result._blocked_navigation = None
    result._blocked_request_count = 0
    return result


def test_fulfilled_response_is_disposed_once_after_body_consumption() -> None:
    owner = browser()
    events: list[str] = []
    response = Response(events)

    owner._guard_navigation(Route(owner, response))

    assert events == [
        "fetch", "headers", "status", "fulfill_consumes_body",
        "fulfill_returned", "dispose",
    ]
    assert response.disposed and response.dispose_attempts == 1
    assert owner._navigation_target_used is True
    assert owner._blocked_request_count == 0


def test_redirect_is_withheld_and_response_disposed_once() -> None:
    owner = browser()
    events: list[str] = []
    response = Response(events, redirect=True)

    owner._guard_navigation(Route(owner, response))

    assert events == ["fetch", "headers", "status", "abort:blockedbyclient", "dispose"]
    assert response.disposed and response.dispose_attempts == 1
    assert owner._blocked_request_count == 1
    assert owner._blocked_navigation == "HTTP redirect requires separate URL approval"


def test_fetch_failure_aborts_without_disposing_unreturned_response() -> None:
    owner = browser()
    events: list[str] = []
    response = Response(events)

    owner._guard_navigation(Route(owner, response, fetch_error=RuntimeError("fetch")))

    assert events == ["fetch", "abort:failed"]
    assert response.dispose_attempts == 0 and not response.disposed


def test_fulfill_failure_is_retained_after_successful_disposal() -> None:
    owner = browser()
    events: list[str] = []
    response = Response(events)
    primary = RuntimeError("fulfill")

    with pytest.raises(RuntimeError) as caught:
        owner._guard_navigation(Route(owner, response, fulfill_error=primary))

    assert caught.value is primary
    assert caught.value.__cause__ is None
    assert events[-2:] == ["fulfill_consumes_body", "dispose"]
    assert response.disposed and response.dispose_attempts == 1


def test_cleanup_failure_after_fulfill_is_propagated() -> None:
    owner = browser()
    events: list[str] = []
    cleanup = RuntimeError("dispose")
    response = Response(events, cleanup_error=cleanup)

    with pytest.raises(RuntimeError) as caught:
        owner._guard_navigation(Route(owner, response))

    assert caught.value is cleanup
    assert events[-2:] == ["fulfill_returned", "dispose"]
    assert response.dispose_attempts == 1 and not response.disposed


def test_dual_failure_preserves_fulfill_error_and_exposes_cleanup_cause() -> None:
    owner = browser()
    events: list[str] = []
    primary = RuntimeError("fulfill")
    cleanup = ValueError("dispose")
    response = Response(events, cleanup_error=cleanup)

    with pytest.raises(RuntimeError) as caught:
        owner._guard_navigation(Route(owner, response, fulfill_error=primary))

    assert caught.value is primary
    assert caught.value.__cause__ is cleanup
    assert response.dispose_attempts == 1 and not response.disposed


@pytest.mark.parametrize("operation", ["headers", "status", "abort"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_navigation_error_remains_primary_when_cleanup_also_fails(
    operation: str, cleanup_fails: bool,
) -> None:
    owner = browser()
    events: list[str] = []
    primary = RuntimeError(operation)
    cleanup = ValueError("dispose") if cleanup_fails else None
    response = Response(
        events,
        redirect=operation == "abort",
        metadata_failure=operation if operation != "abort" else None,
        primary_error=primary,
        cleanup_error=cleanup,
    )
    route = Route(
        owner, response, abort_error=primary if operation == "abort" else None,
    )

    with pytest.raises(RuntimeError) as caught:
        owner._guard_navigation(route)

    assert caught.value is primary
    assert caught.value.__cause__ is cleanup
    assert response.dispose_attempts == 1
    assert response.disposed is (not cleanup_fails)
    assert events[-1] == "dispose"
    assert "fulfill_consumes_body" not in events


def test_non_exception_interruption_also_retains_primary_and_disposes() -> None:
    class Interrupted(BaseException):
        pass

    owner = browser()
    events: list[str] = []
    primary = Interrupted()
    cleanup = RuntimeError("dispose")
    response = Response(events, cleanup_error=cleanup)

    with pytest.raises(Interrupted) as caught:
        owner._guard_navigation(Route(owner, response, fulfill_error=primary))

    assert caught.value is primary
    assert caught.value.__cause__ is cleanup
    assert response.dispose_attempts == 1


@pytest.mark.parametrize("redirect", [False, True])
def test_real_open_surfaces_cleanup_failure_through_existing_guard(
    redirect: bool,
) -> None:
    owner = browser()
    events: list[str] = []
    cleanup = RuntimeError("dispose")
    response = Response(events, redirect=redirect, cleanup_error=cleanup)
    route = Route(owner, response)

    def goto(url: str, **kwargs: object) -> None:
        assert url == route.request.url
        assert kwargs == {"wait_until": "domcontentloaded", "timeout": 30_000}
        owner._guard_navigation(route)

    owner._page.goto = goto
    owner._page.close = lambda: events.append("page_closed")
    owner._ensure_page = lambda: owner._page
    replacement = object()
    owner._context = SimpleNamespace(new_page=lambda: replacement)

    with pytest.raises(PermissionError if redirect else RuntimeError) as caught:
        owner._open(route.request.url)

    if redirect:
        assert caught.value.__cause__ is cleanup
        assert owner._page is replacement
        assert events[-2:] == ["dispose", "page_closed"]
    else:
        assert caught.value is cleanup
        assert "page_closed" not in events
    assert response.dispose_attempts == 1 and not response.disposed
    assert owner._navigation_target is None
    assert owner._navigation_target_used is False
