"""The two highest-trust frontend checks — both PROVEN, both driven against
a real, running Playwright page (never a static parse of the HTML source):

- `dom_check`: the intended change reached the *rendered* DOM — the node
  exists, and carries whatever visibility/class/attribute/text the spec
  expects. Answers "did the change land," nothing about behavior.
- `interaction_check`: perform the actual user action (a real click) and
  assert the actual, observable outcome (URL changed, another element
  appeared) — answers "does the change work," not just "is it present."

Every result is graded strictly by what Playwright reports back — no
heuristics, no "looks about right." That's what keeps these PROVEN.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from verdict.config import DomSpec, InteractionSpec
from verdict.schema import GateStatus, Provenance, Signal

if TYPE_CHECKING:
    from playwright.sync_api import Page

INTERACTION_TIMEOUT_MS = 5_000


def _fail(name: str, detail: str, command: str | None = None) -> Signal:
    return Signal(
        name=name, provenance=Provenance.PROVEN, status=GateStatus.FAIL, detail=detail, command=command
    )


def _pass(name: str, detail: str, command: str | None = None) -> Signal:
    return Signal(
        name=name, provenance=Provenance.PROVEN, status=GateStatus.PASS, detail=detail, command=command
    )


def dom_check(page: Page, spec: DomSpec, check_name: str) -> Signal:
    name = f"frontend:dom:{check_name}"
    command = f"dom: {spec.selector}"
    locator = page.locator(spec.selector)

    if locator.count() == 0:
        return _fail(name, f"no element matched selector {spec.selector!r}", command)

    node = locator.first
    problems: list[str] = []

    if spec.visible is not None:
        actual = node.is_visible()
        if actual != spec.visible:
            problems.append(f"expected visible={spec.visible}, got visible={actual}")

    if spec.class_contains is not None:
        class_attr = node.get_attribute("class") or ""
        if spec.class_contains not in class_attr:
            problems.append(f"class {class_attr!r} does not contain {spec.class_contains!r}")

    if spec.attribute is not None:
        value = node.get_attribute(spec.attribute)
        if value is None:
            problems.append(f"attribute {spec.attribute!r} is not present")
        elif spec.attribute_contains is not None and spec.attribute_contains not in value:
            problems.append(
                f"attribute {spec.attribute!r}={value!r} does not contain "
                f"{spec.attribute_contains!r}"
            )

    if spec.text_contains is not None:
        text = node.inner_text()
        if spec.text_contains not in text:
            problems.append(f"text {text!r} does not contain {spec.text_contains!r}")

    if problems:
        return _fail(name, "; ".join(problems), command)
    return _pass(name, f"{spec.selector} matched the expected DOM state", command)


def interaction_check(page: Page, spec: InteractionSpec, check_name: str) -> Signal:
    name = f"frontend:interaction:{check_name}"
    command = f"click: {spec.click}"
    locator = page.locator(spec.click)

    if locator.count() == 0:
        return _fail(name, f"no element matched selector {spec.click!r} to click", command)

    try:
        locator.first.click(timeout=INTERACTION_TIMEOUT_MS)
    except Exception as exc:  # noqa: BLE001 - Playwright raises its own TimeoutError/Error types
        return _fail(name, f"click on {spec.click!r} failed: {exc}", command)

    problems: list[str] = []

    if spec.expect_url_contains is not None:
        try:
            page.wait_for_url(f"**{spec.expect_url_contains}**", timeout=INTERACTION_TIMEOUT_MS)
        except Exception:  # noqa: BLE001
            problems.append(
                f"expected URL to contain {spec.expect_url_contains!r} after click, "
                f"got {page.url!r}"
            )

    if spec.expect_selector_visible is not None:
        try:
            page.wait_for_selector(
                spec.expect_selector_visible, state="visible", timeout=INTERACTION_TIMEOUT_MS
            )
        except Exception:  # noqa: BLE001
            problems.append(
                f"expected {spec.expect_selector_visible!r} to become visible after click"
            )

    if problems:
        return _fail(name, "; ".join(problems), command)
    return _pass(name, f"clicking {spec.click} produced the expected outcome", command)
