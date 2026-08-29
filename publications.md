---
layout: default
title: Publications
description: Publications, preprints, and selected research outputs.
permalink: /publications/
---

<header class="page-header">
  <p class="eyebrow">Research output</p>
  <h1>Publications</h1>
  <p>Peer-reviewed papers, workshop publications, preprints, and other research outputs.</p>
</header>

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
