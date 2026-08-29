---
layout: default
title: Blog
description: Research notes, technical essays, and ideas in progress.
permalink: /blog/
---

<header class="page-header">
  <p class="eyebrow">Notes &amp; ideas</p>
  <h1>Writing</h1>
  <p>Research notes, technical essays, reading reflections, and ideas in progress.</p>
</header>

<aside class="template-notice" aria-label="Template instructions"><strong>Template content.</strong> Add Markdown files to <code>_posts</code> using the filename <code>YYYY-MM-DD-title.md</code>.</aside>

<div class="post-list">
  {% for post in site.posts %}{% include post-card.html post=post %}{% endfor %}
</div>
