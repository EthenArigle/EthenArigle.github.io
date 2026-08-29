---
layout: default
title: Publications
description: Publications, preprints, and selected research outputs.
permalink: /publications/
---

<header class="page-header">
  <p class="eyebrow">Research output</p>
  <h1>Publications</h1>
  <p>Peer-reviewed papers, preprints, and other research artifacts. The entries below are clearly marked samples and should be replaced with real work.</p>
</header>

<aside class="template-notice" aria-label="Template instructions"><strong>Template content.</strong> Edit <code>_data/publications.yml</code> to manage this list without changing the page layout.</aside>

{% assign sorted_publications = site.data.publications | sort: "year" | reverse %}
{% assign current_year = "" %}
<div class="publication-list">
{% for publication in sorted_publications %}
  {% capture publication_year %}{{ publication.year }}{% endcapture %}
  {% if publication_year != current_year %}
    <h2 class="year-heading">{{ publication.year }}</h2>
    {% assign current_year = publication_year %}
  {% endif %}
  {% include publication.html publication=publication %}
{% endfor %}
</div>
