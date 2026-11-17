const { spawn, execSync } = require('child_process');
const path = require('path');

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

// Start Backend (FastAPI on port 8000 with auto-reload)
// Using single command string with shell: true avoids Node.js DEP0190 warning
// Explicitly watch only backend code and exclude data/store files to avoid file-watcher race conditions on Windows
const backend = spawn('python -m uvicorn backend.main:app --port 8000 --host 127.0.0.1 --reload --reload-dir backend --reload-exclude "demo_data/*" --reload-exclude "*.xlsx" --reload-exclude "*.csv" --reload-exclude "*.pdf"', {
  cwd: __dirname,
  shell: true,
  stdio: 'inherit'
});

const fs = require('fs');

// Start Frontend (auto-detect dev vs production build)
const hasBuild = fs.existsSync(path.join(__dirname, 'frontend', '.next'));
const isProd = process.env.NODE_ENV === 'production';
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

