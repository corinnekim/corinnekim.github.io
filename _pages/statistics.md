---
title: "Statistics"
layout: archive
category: statistics
permalink: /statistics/
author_profile: true  
sidebar:
  nav: "sidebar-category"
---

{% assign posts = site.categories.statistics %}
{% for post in posts %}
  <div class="statistics-item">
    {% include archive-single.html type=page.entries_layout %}
  </div>
{% endfor %}