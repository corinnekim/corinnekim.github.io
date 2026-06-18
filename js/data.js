// 텍스트, 링크, 아이콘, 경로 설정 (언어별로 ko / en 구조)

const I18N = {
  ko: {
    nav: {
      home: "Home",
      education: "Education",
      experience: "Experience",
      projects: "Projects",
      resume: "Resume"
    },
    intro: {
      headline: "안녕하세요, 데이터 사이언티스트 김선우입니다.",
      paragraphs: [
        "저는 복잡한 데이터에서 예상치 못한 패턴을 발견할 때 재미를 느낍니다.",
        "미국 LA 인턴십 당시, SQL로 판매 데이터를 분석해 감으로 하던 발주 결정을 데이터 기반으로 바꾼 경험이 시작이었습니다.",
        "경영학부에서 출발해 패션 산업을 거쳐 데이터 사이언스로 방향을 바꾸고, 네덜란드와 미국에서 낯선 환경에 빠르게 적응해 왔습니다. 선형대수, 통계학, 자료구조 등을 수강하며 수학과 CS 기초를 다졌습니다.",
        "새로운 도메인과 불완전한 데이터 속에서도 스스로 질문을 던지고 끝까지 답을 찾아가는 과정을 즐깁니다."
      ]
    },
    titles: {
      education: "Education",
      experience: "Work Experience",
      projects: "Projects",
      skills: "Skills",
      coursework: "Relevant Coursework"
    },
    education: [
      {
        school: "한양대학교",
        period: "2018.03 - 2024.08",
        location: "Seoul, Korea",
        degree: "경영학부",
        gpa: "GPA 4.27",
        courses: "Marketing Research, Python, AI & Machine Learning"
      },
      {
        school: "The Hague University of Applied Sciences",
        period: "2023.01 - 2023.12",
        location: "The Hague, Netherlands",
        degree: "Exchange Program, International Business"
      }
    ],
    experience: [
      {
        company: "Fashion Debut, Inc.",
        role: "Sales & Marketing Intern",
        period: "2024.07 - 2025.07",
        location: "California, United States",
        bullets: [
          "SQL로 ERP 판매 데이터를 분석해 색상·품목별 수요 지표를 도출하고 리포트로 시각화. 감으로 하던 발주를 데이터 기반 의사결정으로 전환해 전년 대비 매출 약 5% 향상",
          "모델컷 생성, 상세페이지 작성, 라인시트 제작을 AI로 자동화하여 시간 및 비용 20% 이상 절감"
        ]
      }
    ],
    projects: [
      {
        title: "Instacart 고객 세그멘테이션",
        stack: "XGBoost | Customer Segmentation",
        summary: `온라인 장보기 중 엉뚱한 상품이 추천되는 경험에서 출발해, "장바구니에 담는 순서"라는 행동 데이터에서 개인화의 단서를 찾은 프로젝트입니다.`,
        image: "assets/images/groceries.jpg",
        link: "https://sweltering-crane-02b.notion.site/Instacart-7d596d4e0a49827cb37d015c80fe11f8"
      },
      {
        title: "GDELT 위성 촬영 스케줄링 자동화",
        stack: "Kalman Filter | LLM | Anomaly Detection",
        summary: "뉴스 빅데이터에서 위성 촬영 골든타임 안에 어떤 도시를 언제, 어떤 위성으로 찍을지를 자동으로 산출해 의사결정을 5시간 → 15분으로 단축했습니다.",
        image: "assets/images/earth.jpg",
        link: "https://sweltering-crane-02b.notion.site/GDELT-LLM-2ec96d4e0a4983eab5ee818b605b3400"
      },
      {
        title: "MovieLens 영화 추천 시스템",
        stack: "AutoInt+ | Recommendation | Bias Correction",
        summary: "모델이 인기 영화에만 쏠리는 편향을 발견하고, '싫어할 만한 것은 추천하지 않는다'는 철학으로 추천 로직을 설계했습니다.",
        image: "assets/images/netflix2.jpg",
        link: "https://github.com/corinnekim/Movie-Recommendation-System"
      }
    ],
    skills: [
      { label: "Languages", items: "Python, SQL" },
      { label: "Data", items: "Pandas, Google BigQuery, DuckDB, Polars" },
      { label: "ML / Statistics", items: "scikit-learn, ARIMA, Kalman Filter" },
      { label: "LLM", items: "OpenAI·Gemini API, LangChain, RAG, Prompt Engineering" },
      { label: "Tools", items: "Git, Streamlit, Claude Code" }
    ],
    coursework: [
      { courses: "Linear Algebra (A), Statistics for Data Analysis (A)", school: "University of California, San Diego" },
      { courses: "Calculus", school: "Seoul Cyber University", status: "In progress" },
      { courses: "Data Structure, Algorithm", school: "Megazone IT", status: "In progress" }
    ],
    footer: {
      thanks: "감사합니다.",
      rights: "© 2026 Sunwoo Kim",
      backToTop: "Back to top"
    }
  }
};

// 외부 링크 (소셜)
const LINKS = {
  email: "sunwoo090@gmail.com",
  linkedin: "https://www.linkedin.com/in/sun-woo-kim-585964272/",
  github: "https://github.com/corinnekim"
};

// 소셜 아이콘 (인라인 SVG)
const ICONS = {
  email: `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M22 6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6zm-2 0-8 5-8-5h16zm0 12H4V8l8 5 8-5v10z"/></svg>`,
  linkedin: `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M20.45 20.45h-3.56v-5.57c0-1.33-.03-3.04-1.85-3.04-1.86 0-2.14 1.45-2.14 2.94v5.67H9.34V9h3.42v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.07 2.07 0 1 1 0-4.14 2.07 2.07 0 0 1 0 4.14zM7.12 20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.22.79 24 1.77 24h20.45c.98 0 1.78-.78 1.78-1.73V1.73C24 .77 23.2 0 22.22 0z"/></svg>`,
  github: `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.37.5 0 5.78 0 12.29c0 5.2 3.44 9.6 8.21 11.16.6.11.82-.26.82-.58 0-.29-.01-1.05-.02-2.06-3.34.71-4.04-1.6-4.04-1.6-.55-1.38-1.34-1.75-1.34-1.75-1.09-.74.08-.73.08-.73 1.2.08 1.84 1.22 1.84 1.22 1.07 1.8 2.81 1.28 3.5.98.11-.76.42-1.28.76-1.57-2.67-.3-5.47-1.31-5.47-5.84 0-1.29.47-2.34 1.23-3.17-.12-.3-.53-1.52.12-3.16 0 0 1-.32 3.3 1.21a11.5 11.5 0 0 1 6 0c2.28-1.53 3.29-1.21 3.29-1.21.65 1.64.24 2.86.12 3.16.77.83 1.23 1.88 1.23 3.17 0 4.54-2.81 5.53-5.49 5.83.43.37.81 1.1.81 2.22 0 1.6-.01 2.9-.01 3.29 0 .32.22.7.83.58A12.3 12.3 0 0 0 24 12.29C24 5.78 18.63.5 12 .5z"/></svg>`,
  summary: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="9" y1="13" x2="15" y2="13"/><line x1="9" y1="17" x2="15" y2="17"/></svg>`
};

// 이미지 / 파일 경로
const ASSETS = {
  profile: "assets/images/brooklyn_bridge.jpg",
  resume: "assets/sunwookim_resume.pdf?v=2"
};