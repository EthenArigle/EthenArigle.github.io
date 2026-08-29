---
layout: default
title: Gallery
description: "Selected moments and a visual archive beyond Ethen Arigle's research work."
permalink: /gallery/
---

{% assign featured_images = site.static_files | where_exp: "file", "file.path contains '/assets/images/gallery/featured/'" | where_exp: "file", "file.extname != ''" | sort: "path" %}
{% assign archive_images = site.static_files | where_exp: "file", "file.path contains '/assets/images/gallery/archive/'" | where_exp: "file", "file.extname != ''" | sort: "path" %}

<header class="page-header gallery-header">
  <p class="eyebrow">Life beyond the lab</p>
  <h1>Gallery</h1>
  <p>A visual collection of selected moments, places, and everyday inspiration.</p>
</header>

<section class="gallery-block" aria-labelledby="featured-gallery-title">
  <div class="gallery-section-heading">
    <div>
      <p class="eyebrow">Priority display</p>
      <h2 id="featured-gallery-title">Featured moments</h2>
    </div>
    <p>The highlights shown first here and previewed on the homepage.</p>
  </div>
  {% if featured_images.size > 0 %}
    <div class="featured-wall">
      {% for image in featured_images %}
        <figure class="featured-photo">
          <img src="{{ image.path | relative_url }}" alt="Featured gallery image {{ forloop.index }}" loading="lazy" decoding="async">
        </figure>
      {% endfor %}
    </div>
  {% else %}
    <p class="empty-state">Featured photos will appear here.</p>
  {% endif %}
</section>

<section class="gallery-block archive-block" aria-labelledby="archive-gallery-title">
  <div class="gallery-section-heading">
    <div>
      <p class="eyebrow">Kept on display</p>
      <h2 id="archive-gallery-title">Photo archive</h2>
    </div>
    <p>A complete, quieter collection preserved in chronological filename order.</p>
  </div>
  {% if archive_images.size > 0 %}
    <div class="archive-wall">
      {% for image in archive_images %}
        <figure class="archive-photo">
          <img src="{{ image.path | relative_url }}" alt="Archived gallery image {{ forloop.index }}" loading="lazy" decoding="async">
        </figure>
      {% endfor %}
    </div>
  {% else %}
    <p class="empty-state">The archive is ready for future photos.</p>
  {% endif %}
</section>
