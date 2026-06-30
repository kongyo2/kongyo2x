## Documentation Policy

Do not create or expand Markdown docs unless explicitly asked — unprompted, you tend to clutter the repo with over-citation, paternalistic "helpfulness," and runbooks nobody wanted.

## Web Fetch Strategy

When fetching web content, try methods in this order — but if the URL is in a public GitHub repo, `git clone` it instead of fetching files one by one. Move to the next if the current one fails (e.g. 403, timeout, aborted):

1. **WebFetch tool** — Default. Try this first.
2. **curl fallback** — If WebFetch returns 403, retry with `curl -sL -A "claude-code/1.0" <url>`. Many 403s are caused by Cloudflare blocking the default `Claude-User` User-Agent.
3. **readability fallback** — `npx -y @mizchi/readability --format=md "<url>"` extracts the main content (strips nav/ads/sidebars) and serializes to Markdown.

## Code Comments Policy

- Default to zero comments. Code should explain itself through naming and structure
- Do NOT add comments explaining what changed or why (`// changed from X to Y`, `// updated for feature Z` are forbidden)
- Add a comment only as a last resort, when non-obvious logic cannot be clarified by refactoring or renaming first
