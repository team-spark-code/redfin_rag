# Smart Brevity Style News (v1)

[INPUT]
- Title: {{title}}
- URL: {{url}}
- Categories: {{categories}}
- Tags: {{tags}}

[CONTEXT]
{{content}}

[INSTRUCTIONS]
- Return a strict JSON:
  {
    "title": "...",
    "subtitle": "...",
    "tldr": ["...", "..."],
    "body_md": "...",
    "tags": ["..."],
    "sources": ["..."],
    "hero_image_url": null,
    "author_name": null
  }
- Use only the provided context. No hallucinations.
- Keep it concise.
