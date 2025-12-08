---
title: "Blog"
layout: archive
category: blog
permalink: /blog/
author_profile: true  
sidebar:
  nav: "sidebar-category"
---

{% assign posts = site.categories.blog %}
{% for post in posts %}
  <div class="blog-item">
    {% include archive-single.html type=page.entries_layout %}
  </div>
{% endfor %}