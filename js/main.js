// 현재 언어 (기본 ko)
let lang = "ko";

// 점 표기 경로로 값 꺼내기
function getByPath(obj, path) {
  return path.split(".").reduce((acc, key) => (acc ? acc[key] : undefined), obj);
}

// data-i18n 텍스트 채우기
function applyText() {
  const dict = I18N[lang];
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const value = getByPath(dict, el.dataset.i18n);
    if (typeof value === "string") el.textContent = value;
  });
}

// intro 렌더 (문단 / 사진 / 이력서)
function renderIntro() {
  const intro = I18N[lang].intro;

  const box = document.getElementById("intro-paragraphs");
  box.innerHTML = "";
  intro.paragraphs.forEach((text) => {
    const p = document.createElement("p");
    p.textContent = text;
    box.appendChild(p);
  });

  document.getElementById("resume-link").href = ASSETS.resume;
  document.getElementById("profile-img").src = ASSETS.profile;
}

// 소셜 아이콘 렌더 (email / linkedin / github)
function renderSocial() {
  const social = document.getElementById("intro-social");
  social.innerHTML = "";

  const items = [
    { key: "email", href: "mailto:" + LINKS.email },
    { key: "linkedin", href: LINKS.linkedin },
    { key: "github", href: LINKS.github }
  ];

  items.forEach(({ key, href }) => {
    const a = document.createElement("a");
    a.className = "social-link";
    a.href = href;
    a.setAttribute("aria-label", key);
    if (key !== "email") {
      a.target = "_blank";
      a.rel = "noopener";
    }
    a.innerHTML = ICONS[key];
    social.appendChild(a);
  });
}

// Education 렌더 (학교 목록)
function renderEducation() {
  const list = document.getElementById("education-list");
  if (!list) return;
  const items = I18N[lang].education || [];
  list.innerHTML = "";

  items.forEach((e) => {
    const entry = document.createElement("div");
    entry.className = "entry";

    const main = document.createElement("div");
    main.className = "entry-main";
    const school = document.createElement("div");
    school.className = "entry-school";
    school.textContent = e.school;
    main.appendChild(school);

    if (e.degree) {
      const degree = document.createElement("div");
      degree.className = "entry-degree";
      degree.textContent = e.degree;
      if (e.gpa) {
        const gpa = document.createElement("span");
        gpa.className = "entry-gpa";
        gpa.textContent = e.gpa;
        degree.appendChild(gpa);
      }
      main.appendChild(degree);
    }
    entry.appendChild(main);

    const side = document.createElement("div");
    side.className = "entry-side";
    const period = document.createElement("div");
    period.className = "entry-period";
    period.textContent = e.period;
    side.appendChild(period);
    if (e.location) {
      const loc = document.createElement("div");
      loc.className = "entry-location";
      loc.textContent = e.location;
      side.appendChild(loc);
    }
    entry.appendChild(side);

    if (e.courses) {
      const detail = document.createElement("div");
      detail.className = "entry-detail";
      detail.textContent = "Courses: " + e.courses;
      entry.appendChild(detail);
    }

    list.appendChild(entry);
  });
}

// Experience 렌더 (경력 목록)
function renderExperience() {
  const list = document.getElementById("experience-list");
  if (!list) return;
  const items = I18N[lang].experience || [];
  list.innerHTML = "";

  items.forEach((x) => {
    const entry = document.createElement("div");
    entry.className = "entry";

    const main = document.createElement("div");
    main.className = "entry-main";
    const company = document.createElement("div");
    company.className = "entry-school";
    company.textContent = x.company;
    main.appendChild(company);
    if (x.role) {
      const role = document.createElement("div");
      role.className = "entry-degree";
      role.textContent = x.role;
      main.appendChild(role);
    }
    entry.appendChild(main);

    const side = document.createElement("div");
    side.className = "entry-side";
    const period = document.createElement("div");
    period.className = "entry-period";
    period.textContent = x.period;
    side.appendChild(period);
    if (x.location) {
      const loc = document.createElement("div");
      loc.className = "entry-location";
      loc.textContent = x.location;
      side.appendChild(loc);
    }
    entry.appendChild(side);

    if (x.bullets && x.bullets.length) {
      const ul = document.createElement("ul");
      ul.className = "entry-bullets";
      x.bullets.forEach((b) => {
        const li = document.createElement("li");
        li.textContent = b;
        ul.appendChild(li);
      });
      entry.appendChild(ul);
    }

    list.appendChild(entry);
  });
}

// Projects 렌더 (프로젝트 카드)
function renderProjects() {
  const list = document.getElementById("projects-list");
  if (!list) return;
  const items = I18N[lang].projects || [];
  list.innerHTML = "";

  items.forEach((p) => {
    const card = document.createElement("div");
    card.className = "project";

    const text = document.createElement("div");
    text.className = "project-text";

    const title = document.createElement("h3");
    title.className = "project-title";
    title.textContent = p.title;
    text.appendChild(title);

    const stack = document.createElement("div");
    stack.className = "project-stack";
    stack.textContent = p.stack;
    text.appendChild(stack);

    const desc = document.createElement("p");
    desc.className = "project-desc";
    desc.textContent = p.summary;
    text.appendChild(desc);

    const link = document.createElement("a");
    link.className = "project-link";
    link.href = p.link;
    link.target = "_blank";
    link.rel = "noopener";
    link.innerHTML = ICONS.summary + "<span>Project Summary</span>";
    text.appendChild(link);

    card.appendChild(text);

    const imageLink = document.createElement("a");
    imageLink.className = "project-image";
    imageLink.href = p.link;
    imageLink.target = "_blank";
    imageLink.rel = "noopener";
    imageLink.setAttribute("aria-label", p.title);
    const img = document.createElement("img");
    img.src = p.image;
    img.alt = p.title;
    imageLink.appendChild(img);
    card.appendChild(imageLink);

    list.appendChild(card);
  });
}

// Skills 렌더 (카테고리별 나열)
function renderSkills() {
  const list = document.getElementById("skills-list");
  if (!list) return;
  const items = I18N[lang].skills || [];
  list.innerHTML = "";

  items.forEach((s) => {
    const row = document.createElement("div");
    row.className = "skill-row";

    const label = document.createElement("div");
    label.className = "skill-label";
    label.textContent = s.label;

    const vals = document.createElement("div");
    vals.className = "skill-items";
    s.items.split(", ").forEach((item) => {
      const span = document.createElement("span");
      span.textContent = item;
      vals.appendChild(span);
    });

    row.append(label, vals);
    list.appendChild(row);
  });
}

// 모바일 햄버거 메뉴
function setupMobileMenu() {
  const toggle = document.getElementById("nav-toggle");
  const links = document.getElementById("nav-links");

  toggle.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  });

  links.querySelectorAll("a").forEach((a) =>
    a.addEventListener("click", () => {
      links.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    })
  );
}

// 한/영 토글
function setupLangToggle() {
  const btn = document.getElementById("lang-toggle");
  btn.textContent = lang === "ko" ? "EN" : "KO";

  btn.addEventListener("click", () => {
    const next = lang === "ko" ? "en" : "ko";
    if (!I18N[next]) return;
    lang = next;
    btn.textContent = lang === "ko" ? "EN" : "KO";
    applyText();
    renderIntro();
    renderEducation();
    renderExperience();
    renderProjects();
    renderSkills();
  });
}

applyText();
renderIntro();
renderSocial();
renderEducation();
renderExperience();
renderProjects();
renderSkills();
setupMobileMenu();
setupLangToggle();