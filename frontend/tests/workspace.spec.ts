import { test, expect } from "@playwright/test";

test("workspace, views, inspector and all navigation states", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByText("Engine connected")).toBeVisible();
  await expect(page.locator(".three-scene canvas")).toBeVisible();
  await page.screenshot({
    path: "../artifacts/web-workspace-1440.png",
    fullPage: true,
  });
  for (const view of ["Top", "Front", "Side", "Perspective"]) {
    await page.getByRole("button", { name: view, exact: true }).click();
    await expect(
      page.getByRole("button", { name: view, exact: true }),
    ).toHaveAttribute("aria-pressed", "true");
  }
  await page.getByRole("button", { name: "Right arm", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Right arm", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  for (const label of ["Calibration", "Session", "Diagnostics"]) {
    await page.getByRole("button", { name: label, exact: true }).click();
    await page.screenshot({
      path: `../artifacts/web-${label.toLowerCase()}.png`,
      fullPage: true,
    });
  }
  await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await page.getByRole("button", { name: "Preferences", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("checkbox", { name: /Floor grid/ }).uncheck();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.reload();
  await page.getByRole("button", { name: "Preferences", exact: true }).click();
  await expect(
    page.getByRole("checkbox", { name: /Floor grid/ }),
  ).not.toBeChecked();
  await page.getByRole("checkbox", { name: /Floor grid/ }).check();
  await page.keyboard.press("Escape");
  await page.keyboard.press("c");
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.screenshot({ path: "../artifacts/web-calibration-dialog.png" });
  await page.keyboard.press("Escape");
  await page.keyboard.press("v");
  await expect(page.locator(".workspace")).toHaveClass(/focused/);
  await page.keyboard.press("v");
  expect(errors).toEqual([]);
});

for (const [width, height] of [
  [1280, 720],
  [1024, 768],
  [760, 900],
  [390, 844],
]) {
  test(`layout at ${width}x${height}`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await page.goto("/");
    await expect(page.locator(".three-scene canvas")).toBeVisible();
    await expect(page.getByText("Engine connected")).toBeAttached();
    await page.screenshot({
      path: `../artifacts/web-workspace-${width}.png`,
      fullPage: true,
    });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
  });
}

test("offline and malformed telemetry states are explicit", async ({
  page,
}) => {
  await page.routeWebSocket("**/api/v1/telemetry", (socket) => {
    socket.send(JSON.stringify({ schema_version: 77 }));
  });
  await page.goto("/");
  await expect(page.getByRole("status").first()).toContainText(
    "unsupported state",
  );
  await expect(
    page.getByRole("button", { name: "Record", exact: true }),
  ).toBeDisabled();
});
