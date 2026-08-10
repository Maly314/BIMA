const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const { execFileSync } = require("node:child_process");
const { pathToFileURL } = require("node:url");

const root = path.resolve(__dirname, "..");

test("Vinext production cache resolves browser asset URLs on Windows", async () => {
  execFileSync(process.execPath, [path.join(root, "scripts/patch-vinext-windows-assets.cjs")], { cwd: root });
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "bima-vinext-assets-"));
  try {
    const assets = path.join(temp, "assets");
    fs.mkdirSync(assets);
    fs.writeFileSync(path.join(assets, "app-test123.css"), "body{color:teal}");
    const moduleUrl = pathToFileURL(path.join(root, "node_modules/vinext/dist/server/static-file-cache.js")).href;
    const { StaticFileCache } = await import(`${moduleUrl}?bima=${Date.now()}`);
    const cache = await StaticFileCache.create(temp);
    const entry = cache.lookup("/assets/app-test123.css");
    assert.ok(entry, "forward-slash browser URL must match the Windows filesystem entry");
    assert.equal(entry.original.headers["Content-Type"], "text/css");
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
});
