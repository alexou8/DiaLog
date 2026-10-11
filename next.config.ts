import type { NextConfig } from 'next';

/** Security headers applied to every response. */
const securityHeaders = [
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'X-Frame-Options', value: 'DENY' },
  { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
  { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
  { key: 'Strict-Transport-Security', value: 'max-age=63072000; includeSubDomains; preload' },
  {
    key: 'Content-Security-Policy',
    value: [
      "default-src 'self'",
      // Next.js injects inline bootstrap scripts; styles are Tailwind-generated + inline chart vars.
      //
      // Known gap: 'unsafe-inline' in script-src means this CSP does not stop
      // an injected inline <script> — most of its XSS value is forfeit, and
      // React's output escaping is doing that job alone. The fix is a per-
      // request nonce set in middleware with 'strict-dynamic' (supported by
      // Next 15), which also makes every page dynamically rendered. Deferred,
      // not forgotten; see docs/SECURITY.md, "CSP and security headers".
      "script-src 'self' 'unsafe-inline'" +
        (process.env.NODE_ENV === 'development' ? " 'unsafe-eval'" : ''),
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "font-src 'self' data:",
      "connect-src 'self'",
      "form-action 'self'",
      "frame-ancestors 'none'",
      "base-uri 'self'",
      "object-src 'none'",
    ].join('; '),
  },
];

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  serverExternalPackages: ['bcryptjs', 'exceljs'],
  experimental: {
    serverActions: {
      // The import upload is a Server Action, and Next's default body limit is
      // 1 MB, so a real Apple Health or LibreView export failed with a bare
      // framework error before the import code could explain anything. 20 MB
      // matches the import file ceiling in lib/import/limits.ts — kept as a
      // literal rather than imported so this config stays free of app code.
      // If one changes, change the other, or files between the two sizes get
      // the framework error back. This also bounds every other Server Action.
      // Set 1 MB above the ceiling because the limit counts the multipart
      // envelope too: at exactly '20mb' a file within a few KB of 20 MB was
      // still refused with the bare framework error. One caveat remains: the
      // hosting platform may cap request bodies lower than this (Vercel
      // Functions document 4.5 MB), in which case that cap wins.
      bodySizeLimit: '21mb',
    },
  },
  async headers() {
    return [
      { source: '/:path*', headers: securityHeaders },
      // Never let authenticated app surfaces be indexed.
      {
        source: '/app/:path*',
        headers: [{ key: 'X-Robots-Tag', value: 'noindex, nofollow, noarchive' }],
      },
    ];
  },
};

export default nextConfig;
