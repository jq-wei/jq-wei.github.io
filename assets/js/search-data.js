// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "dropdown-bookshelf",
              title: "bookshelf",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/books/";
              },
            },{id: "dropdown-blog",
              title: "blog",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/blog/";
              },
            },{id: "post-",
        
          title: "",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/2026-01-08-mhc-manifold-constrained-hyper-connections/";
          
        },
      },{id: "post-react-synergizing-reasoning-and-acting-in-language-models",
        
          title: "ReAct: Synergizing Reasoning and Acting in Language Models",
        
        description: "Notes on ReAct - a paradigm for LLM agents that combines reasoning traces and task-specific actions. Covers prompt engineering, fine-tuning approaches, and implementation of zero-shot ReAct agents with LangChain.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/react-synergizing-reasoning-acting/";
          
        },
      },{id: "post-a-survey-on-in-context-learning",
        
          title: "A Survey on In-Context Learning",
        
        description: "Survey notes on in-context learning (ICL) - a paradigm for LLMs to learn tasks from few examples without parameter updates. Covers ICL mechanisms, prompt engineering, and comparison of causalLM vs prefixLM.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/in-context-learning-survey/";
          
        },
      },{id: "post-transformer-basics-architecture-and-data-flow",
        
          title: "Transformer Basics: Architecture and Data Flow",
        
        description: "Understanding the transformer architecture from model layers and token sequence perspectives",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/transformer-basics/";
          
        },
      },{id: "post-hipporag-neurobiologically-inspired-long-term-memory-for-llms",
        
          title: "HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs",
        
        description: "Paper reading notes on HippoRAG - using knowledge graphs and PageRank for better RAG retrieval",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/hipporag-paper-notes/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-main-topics-i-worked-on-in-hkrc-huawei",
          title: 'Main topics I worked on in HKRC Huawei',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/announcement_2/";
            },},{id: "news-i-will-join-eqt-group-to-develop-ai-solutions-in-finance",
          title: 'I will join EQT group to develop AI solutions in Finance.',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6A%69%65%71%69%61%6E%67.%77%65%69.%64%65%76@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/jq-wei", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/jieqiang-wei", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=5ClEDSoAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://patents.google.com/?inventor=jieqiang+wei", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
