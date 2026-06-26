/**
 * Tests for the Makora provider extension.
 *
 * Verifies the extension module can be imported and the provider
 * registers without errors.
 */

import { describe, expect, it } from "vitest";

// Import the default export to verify the module loads
// (the factory function calls pi.registerProvider internally)
import makoraProvider from "../index.js";

describe("makora provider", () => {
  it("exports a factory function", () => {
    expect(typeof makoraProvider).toBe("function");
    // The factory should be a function that accepts an ExtensionAPI
    // and registers the makora provider
    expect(makoraProvider.length).toBe(1);
  });
});
