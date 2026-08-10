/* Vinext 0.0.50 stores path.relative() results directly in its production
   static cache. On Windows those keys contain backslashes, while browser URLs
   use forward slashes, so every /assets/*.css and /assets/*.js request is a
   cache miss. Keep this small compatibility patch until the dependency ships
   the same normalization upstream. */
const fs = require("node:fs");
const path = require("node:path");

const target = path.resolve(__dirname, "../node_modules/vinext/dist/server/static-file-cache.js");
if (!fs.existsSync(target)) {
  throw new Error(`Vinext production server was not found at ${target}`);
}

const before = "relativePath: path.relative(base, batch[j]),";
const after = 'relativePath: path.relative(base, batch[j]).split(path.sep).join("/"),';
const source = fs.readFileSync(target, "utf8");
if (source.includes(after)) {
  process.stdout.write("Vinext Windows asset-path patch already applied.\n");
} else if (source.includes(before)) {
  fs.writeFileSync(target, source.replace(before, after), "utf8");
  process.stdout.write("Applied Vinext Windows asset-path patch.\n");
} else {
  throw new Error("Vinext static-cache implementation changed; refusing an unsafe patch.");
}
