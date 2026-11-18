const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log("=================================================");
console.log(" Starting InsightFlow AI Unified Single Server");
console.log(" Working Link: http://localhost:8000");
console.log(" Frontend Link: http://localhost:3000");
console.log("=================================================");

// Ensure ports 8000 and 3000 are clean before starting on Windows
try {
  if (process.platform === 'win32') {
    execSync('powershell -NoProfile -Command "Get-Process -Id (Get-NetTCPConnection -LocalPort 3000, 8000 -ErrorAction SilentlyContinue).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue"', { stdio: 'ignore' });
  }
} catch (e) {}

const port = process.env.PORT || 8000;
const isProd = process.env.NODE_ENV === 'production';

// Start Backend (FastAPI with auto-reload in dev, single process in prod)
// Explicitly reload only backend code directory and exclude data/store files without shell wildcard expansion
const uvicornCmd = isProd
  ? `python -m uvicorn backend.main:app --port ${port} --host 0.0.0.0`
  : `python -m uvicorn backend.main:app --port ${port} --host 127.0.0.1 --reload --reload-dir backend --reload-exclude demo_data/*`;

const backend = spawn(uvicornCmd, {
  cwd: __dirname,
  shell: true,
  stdio: 'inherit'
});

// Start Frontend (auto-detect dev vs production build)
const hasBuild = fs.existsSync(path.join(__dirname, 'frontend', '.next'));
const frontendCmd = (isProd && hasBuild) ? 'npx next start -p 3000' : 'npx next dev -p 3000';

const frontend = spawn(frontendCmd, {
  cwd: path.join(__dirname, 'frontend'),
  shell: true,
  stdio: 'inherit'
});

const cleanup = () => {
  console.log("\nShutting down InsightFlow AI servers...");
  try {
    if (process.platform === 'win32') {
      if (backend.pid) execSync(`taskkill /pid ${backend.pid} /T /F 2>nul`, { stdio: 'ignore' });
      if (frontend.pid) execSync(`taskkill /pid ${frontend.pid} /T /F 2>nul`, { stdio: 'ignore' });
      execSync('powershell -NoProfile -Command "Get-Process -Id (Get-NetTCPConnection -LocalPort 3000, 8000 -ErrorAction SilentlyContinue).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue"', { stdio: 'ignore' });
    } else {
      backend.kill();
      frontend.kill();
    }
  } catch (e) {}
  process.exit();
};

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);
