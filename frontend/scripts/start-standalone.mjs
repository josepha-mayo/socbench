import { cpSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const standaloneRoot = join(root, ".next", "standalone");
const serverPath = join(standaloneRoot, "server.js");
const staticSource = join(root, ".next", "static");
const staticTarget = join(standaloneRoot, ".next", "static");

function optionValue(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

if (!existsSync(serverPath) || !existsSync(staticSource)) {
  throw new Error("Standalone assets are missing. Run `npm run build` before `npm run start`.");
}

cpSync(staticSource, staticTarget, { recursive: true, force: true });
process.env.PORT ||= optionValue("--port") || "3000";
process.env.HOSTNAME ||= optionValue("--hostname") || "127.0.0.1";

await import(pathToFileURL(serverPath).href);
