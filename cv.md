---
layout: default
title: CV
description: Curriculum vitae, academic experience, honors, and service.
permalink: /cv/
---

<header class="page-header cv-header">
  <div><p class="eyebrow">Curriculum vitae</p><h1>{{ site.data.profile.name }}</h1><p>{{ site.data.profile.role }} · {{ site.data.profile.institution }}</p></div>
  {% if site.data.profile.cv != "" %}<a class="button primary" href="{{ site.data.profile.cv | relative_url }}">Download PDF</a>{% else %}<span class="button disabled" aria-disabled="true">PDF coming soon</span>{% endif %}
</header>

<div class="cv-layout">
  <aside class="cv-aside" aria-label="Research interests">
    <h2>Research interests</h2>
    <ul>{% for interest in site.data.profile.interests %}<li>{{ interest.name }}</li>{% endfor %}</ul>
  </aside>
  <div class="cv-main">
    <section class="cv-section">
      <h2>Education</h2>
      {% for education in site.data.profile.education %}
        <div class="cv-item">
          <time>{{ education.period }}</time>
          <div><h3>{{ education.degree }}</h3><p>{{ education.school }}</p><p>{{ education.department }}</p></div>
        </div>
      {% endfor %}
    </section>
    <section class="cv-section">
      <h2>Research profile</h2>
      <div class="cv-item"><time>ORCID</time><div><h3><a href="{{ site.data.profile.links.orcid }}">0009-0002-3954-9095</a></h3><p>Persistent researcher identifier and publication record.</p></div></div>
      <div class="cv-item"><time>Contact</time><div><h3><a href="mailto:{{ site.data.profile.email }}">{{ site.data.profile.email }}</a></h3><p>Academic correspondence.</p></div></div>
    </section>
  </div>
</div>
