import type { NextConfig } from "next";

let rawBackend = (
  process.env.BACKEND_API_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000"
).trim().replace(/\/+$/, "");

if (!rawBackend.startsWith("http://") && !rawBackend.startsWith("https://")) {
  rawBackend = `https://${rawBackend}`;
}

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${rawBackend}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
