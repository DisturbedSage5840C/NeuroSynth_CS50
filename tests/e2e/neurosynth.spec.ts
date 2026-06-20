import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

async function assertNoCriticalA11yViolations(page: any) {
  const results = await new AxeBuilder({ page }).analyze();
  const critical = results.violations.filter((v) => v.impact === "critical");
  expect(critical, `Critical accessibility violations: ${JSON.stringify(critical, null, 2)}`).toHaveLength(0);
}

test.describe("NeuroSynth clinical workflow", () => {
  test("login -> patient -> connectome -> report -> uncertainty badge", async ({ page }) => {
    await page.goto("/");

    // Login flow
    await page.getByLabel("Username").fill("clinician.qa");
    await page.getByLabel("Password").fill("test-password");
    await page.getByRole("button", { name: /sign in|login/i }).click();

    // Patient list should render quickly for clinical workflow.
    const started = Date.now();
    await expect(page.getByText(/patient/i).first()).toBeVisible();
    const elapsedMs = Date.now() - started;
    expect(elapsedMs).toBeLessThan(2000);

    await assertNoCriticalA11yViolations(page);

    // Click first patient row/card.
    const patientRow = page.locator("[data-testid='patient-item'], .patient-item, [role='row']").first();
    await patientRow.click();

    // Connectome graph is expected to expose SVG nodes.
    const graph = page.locator("svg").first();
    await expect(graph).toBeVisible();
    const nodeCount = await graph.locator("circle").count();
    expect(nodeCount).toBeGreaterThan(0);

    // Expand report section.
    const reportToggle = page.getByRole("button", { name: /report|evidence|section/i }).first();
    await reportToggle.click();

    // Assert uncertainty badge visibility.
    await expect(page.getByText(/CI\s*\d+%|confidence|uncertainty/i).first()).toBeVisible();

    await assertNoCriticalA11yViolations(page);
  });
});
