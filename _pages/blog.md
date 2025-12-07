---
title: "Blog"
layout: archive
permalink: "/blog/"
categories: [blog]
---

{% assign posts = site.categories.blog %}
{% for post in posts %}
  <div class="blog-item">
    {% include archive-single.html type=page.entries_layout %}
  </div>
{% endfor %}