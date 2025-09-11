# how to use

1. make the modifications and push. Github will automatically compile and deploy which takes about 5 mins. 

# Project pages

1. The main page is `_pages/projects.md` which iterates all the pages (md files) defined in `_projects/*.md` 

2. For now, this page is disabled.

# Publication pages

1. this page is disabled for now. 
2. To turn it back on, change `_page/publications.md` from 
```markdown
nav: false #true
#nav_order: 2
```
to 
```markdown
nav: true
nav_order: 2
```