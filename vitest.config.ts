import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  test: {
    environment: "node",
    include: ["__tests__/**/*.test.ts"],
    clearMocks: true,
    restoreMocks: true,
  },
  resolve: {
    alias: {
      // index.ts imports streamOpenAICompletions from the /compat subpath and
      // clampThinkingLevel + types from the main entry. Without mocking these,
      // tests would make real HTTP calls. Alias both paths to one stub that
      // records streamOpenAICompletions calls and lets tests override clamp.
      "@earendil-works/pi-ai/compat": path.resolve(__dirname, "__tests__/__mocks__/pi-ai.ts"),
      "@earendil-works/pi-ai": path.resolve(__dirname, "__tests__/__mocks__/pi-ai.ts"),
    },
  },
});
