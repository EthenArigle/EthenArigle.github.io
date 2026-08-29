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

<aside class="template-notice" aria-label="Template instructions"><strong>Template content.</strong> Replace every bracketed item and add a PDF path in <code>_data/profile.yml</code> when ready.</aside>

<div class="cv-layout">
  <aside class="cv-aside" aria-label="Research interests">
    <h2>Research interests</h2>
    <ul>{% for interest in site.data.profile.interests %}<li>{{ interest.name }}</li>{% endfor %}</ul>
  </aside>
  <div class="cv-main">
    <section class="cv-section"><h2>Education</h2><div class="cv-item"><time>[20XX–20XX]</time><div><h3>[Degree in Field]</h3><p>[University], [City, Country]</p><p>Advisor: [Advisor Name]</p></div></div></section>
    <section class="cv-section"><h2>Experience</h2><div class="cv-item"><time>[20XX–Present]</time><div><h3>[Academic or Research Position]</h3><p>[Institution or Laboratory]</p></div></div></section>
    <section class="cv-section"><h2>Honors</h2><div class="cv-item"><time>[20XX]</time><div><h3>[Award or Fellowship]</h3><p>[Awarding organization]</p></div></div></section>
    <section class="cv-section"><h2>Academic service</h2><div class="cv-item"><time>[20XX]</time><div><h3>[Reviewer, Organizer, or Volunteer Role]</h3><p>[Conference, journal, or community]</p></div></div></section>
  </div>
</div>
