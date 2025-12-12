# Portfolio Website Project Context

> **For Claude Code**: This file contains all context needed to build this portfolio website.
> Read this entire file before starting any implementation.

---

## Project Goal

Build an interactive portfolio website for **Shammunul Islam** featuring:
- Professional portfolio sections (About, Research, Projects, CV, Contact)
- **Interactive AI paper explanation blogs** with real-time code visualization
- Admin dashboard for content management
- Clean, academic, modern design

---

## Site Sections Overview

| Section | Description | Priority | Status |
|---------|-------------|----------|--------|
| Home | Landing page, intro, highlights | Must have | To build |
| About | Bio, education, skills | Must have | To build |
| AI Paper Implementations | Interactive blog series (CORE FEATURE) | Must have | Content ready |
| Research & Publications | Papers, thesis, books | Must have | To build |
| Projects | GitHub projects showcase | Should have | To build |
| CV | Downloadable resume | Should have | PDF ready |
| Contact | Contact form, social links | Must have | To build |
| Blog (general) | Non-paper posts | Nice to have | Future |

### Section Details

#### 1. Home
- Hero section with name, title, tagline
- Brief intro paragraph
- Featured content cards (latest blog, key project, recent publication)
- Quick navigation to main sections
- Professional but welcoming tone

#### 2. About
- Professional bio (from /personal/bio.md)
- Research interests with visual tags/icons
- Education timeline (interactive if possible)
- Technical skills grouped by category
- Profile photo

#### 3. AI Paper Implementations from Scratch ⭐ (CORE FEATURE)
**This is the main differentiating feature of the site.**

- Landing page listing all paper series
- Each paper is a "series" with multiple "chunks"
- Each chunk has:
  - Theoretical explanation (markdown with LaTeX math)
  - Python code blocks
  - Interactive code editor (users can modify)
  - Real-time visualization output
  - Reset/Copy/Download buttons for code
- Progress tracking (optional: users see completed chunks)
- Difficulty indicators
- Prerequisites listed
- Estimated reading time

---

## AI Paper Implementations - Content Plan

### Currently Implementing ✅

| Paper | Authors | Venue | Year | ArXiv | Status |
|-------|---------|-------|------|-------|--------|
| **Fourier Neural Operator for Parametric PDEs** | Li et al. | ICLR | 2021 | [2010.08895](https://arxiv.org/abs/2010.08895) | Content Ready |

**GitHub**: [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)

**Tutorial Structure (6 Chunks):**
| Chunk | Title | Estimated Time |
|-------|-------|----------------|
| 1 | Mathematical Foundations | 45 min |
| 2 | Fourier Theory for Neural Networks | 60 min |
| 3 | Neural Operator Framework | 45 min |
| 4 | FNO Architecture | 90 min |
| 5 | Training Methodology | 60 min |
| 6 | Applications & Benchmarks | 60 min |

**Total**: ~6 hours of content

---

### Planned Future Papers 📋

| Priority | Paper | Authors | Year | ArXiv |
|----------|-------|---------|------|-------|
| 🔴 High | **Physics-Informed Neural Networks (PINNs)** | Raissi, Perdikaris & Karniadakis | 2019 | 1711.10561 |
| 🔴 High | **Neural Ordinary Differential Equations** | Chen et al. | 2018 | 1806.07366 |
| 🔴 High | **Pangu-Weather** | Bi et al. | 2023 | 2211.02556 |
| 🔴 High | **FourCastNet** | Pathak et al. | 2022 | 2202.11214 |
| 🔴 High | **ClimaX** | Nguyen et al. | 2023 | 2301.10343 |
| 🟡 Medium | **DeepONet: Learning Nonlinear Operators** | Lu et al. | 2021 | 1910.03193 |
| 🟡 Medium | **Attention Is All You Need (Transformers)** | Vaswani et al. | 2017 | 1706.03762 |
| 🟡 Medium | **Vision Transformer (ViT)** | Dosovitskiy et al. | 2020 | 2010.11929 |
| 🟡 Medium | **GraphCast** | Lam et al. | 2023 | 2212.12794 |

**Focus Areas:**
- Physics-Informed ML (PINNs, Neural Operators)
- Foundation Models for Climate/Weather
- Transformers and Attention Mechanisms

---

#### 4. Research & Publications
- Peer-reviewed publications (3 papers)
- Books authored (2 books on GIS/Remote Sensing)
- MS Thesis information
- Links to Google Scholar, ORCID
- Citation counts if available
- PDF downloads where permitted

#### 5. Projects
- GitHub repositories showcase
- Project cards with:
  - Title and description
  - Tech stack badges
  - GitHub link
  - Live demo link (if applicable)
  - Screenshot/thumbnail

#### 6. CV
- Interactive timeline view (optional)
- Download PDF button (prominent)
- Key highlights visible on page
- Last updated date

#### 7. Contact
- Contact form (email integration)
- Direct email link
- Social links: LinkedIn, GitHub, Google Scholar, ORCID, Twitter/X
- Location (general: Virginia, USA)
- Open to: collaborations, PhD opportunities

#### 8. Blog (General) - Future
- Implementations of AI papers from scratch
- GeoAI and Climate Change
- AI in Remote Sensing
- Book discussions
- Non-paper tutorials and posts
- Can be added later
- Lower priority

---

## Admin Dashboard Requirements

### Authentication
- Secure admin login (single user - site owner only)
- Password protected with secure session
- Consider: NextAuth, Supabase Auth, or similar

### Dashboard Features

#### Content Management
| Feature | Description |
|---------|-------------|
| Blog Series CRUD | Add/Edit/Delete paper series |
| Chunk CRUD | Add/Edit/Delete chunks within series |
| Markdown Editor | Rich editor for theory content with preview |
| Code Editor | Syntax-highlighted editor for Python files |
| Image Upload | Upload figures, diagrams |
| Ordering | Drag-and-drop to reorder chunks |
| Metadata | Edit title, description, difficulty, prerequisites |

#### Draft/Publish System
- Save content as draft
- Preview before publishing
- Publish/Unpublish toggle
- Schedule publishing (nice to have)

#### Site Content Management
- Edit About page content
- Update publications list
- Modify projects
- Update CV file

#### User Interactions (if implemented)
- View comments
- Moderate/delete inappropriate comments
- View basic analytics (page views)

### Admin UI
- Clean, simple interface
- Mobile-friendly (for quick edits on phone)
- Accessible from /admin route (hidden from main nav)

---

## Interactive Code Features (Critical for Blog Section)

### Code Editor Component
- Syntax highlighting for Python
- Line numbers
- Dark/light theme matching site
- Resizable editor area
- Auto-indentation
- Error highlighting (if possible)

### Code Execution Engine
- Run Python code in browser using Pyodide (CPython in WebAssembly)
- Required libraries: numpy, scipy, matplotlib, (pandas if needed)
- Display:
  - Print/console output
  - Matplotlib figures as images
  - Error messages with helpful formatting
- Execution timeout (prevent infinite loops)
- Memory limits

### User Interactions with Code
| Button | Action |
|--------|--------|
| ▶ Run | Execute the code |
| ↺ Reset | Restore original code |
| 📋 Copy | Copy code to clipboard |
| ⬇ Download | Download as .py file |
| 🔗 Share | (Optional) Share modified version |

### Visualization Updates
- When user modifies code and runs, visualization updates
- Side-by-side or stacked layout (theory | code | output)
- Responsive: stack on mobile, side-by-side on desktop

### Multiple Code Blocks
- Each chunk can have multiple code blocks
- Each runs independently
- Option to "Run All" sequentially
- Variables persist within a chunk session (if feasible)

---

## Owner Information

### Personal Details
- **Name**: Shammunul Islam
- **Current Role**: MS Climate Science student, George Mason University
- **Thesis**: Predicting Land Surface Temperature using Machine Learning and Satellite Remote Sensing
- **Thesis Defense**: November 2025
- **Location**: Virginia, USA

### Contact
- **Email**: sha_is13@yahoo.com / sislam27@mason.edu
- **LinkedIn**: [linkedin.com/in/shammunul](https://www.linkedin.com/in/shammunul/)
- **GitHub**: [github.com/shammun](https://github.com/shammun)

### Academic Background
- MS Climate Science, George Mason University (2025)
- MA Climate and Society, Columbia University (2014) - Full scholarship
- BSc Statistics, Shahjalal University of Science & Technology

### Professional Experience
- Local Climate Action Fellow, Virginia Climate Center (2024-Present)
- Research Assistant/Atmospheric Modeler, George Mason University (2023-2024)
- Adjunct Faculty, Jahangirnagar University (2021-2023)
- Country Consultant, IRI Columbia University (2019-2022)
- Spatial Data Scientist, Consultant
- Survey Statistician

### Publications & Books
- 3 peer-reviewed publications
- 2 books on GIS and Remote Sensing (Packt Publishing)
- AGU25 poster presentation
- See /personal/publications.json for details

### Research Interests
- Physics-Informed Machine Learning for Climate
- Urban Heat Island Prediction and Mapping
- Statistical and Dynamical Climate Downscaling
- Neural Operators for PDEs (FNO, DeepONet)
- Foundation Models for Climate Science
- AI in Remote Sensing
- Satellite Remote Sensing (ECOSTRESS, Landsat, MODIS)

---

## Design Philosophy & Visual Identity

> **Core Principle**: The design should reflect the intersection of **climate science**, **statistical rigor**, and **modern AI/ML** — communicating precision, clarity, and scientific credibility while remaining approachable and visually engaging.

### Design DNA

The website's visual identity draws from three core themes:

1. **Climate & Earth Systems** — Subtle nods to atmospheric data, satellite imagery aesthetics, temperature gradients, and environmental patterns
2. **Statistical Precision** — Clean grids, data-driven layouts, mathematical elegance, and information hierarchy
3. **AI/ML Innovation** — Modern, tech-forward elements, neural network-inspired patterns, and interactive dynamism

### Visual Metaphors to Incorporate

| Theme | Visual Expression |
|-------|-------------------|
| Climate Science | Gradient backgrounds reminiscent of temperature maps, subtle topographic or contour line patterns, earth/ocean color influences |
| Statistics | Grid-based layouts, clean data tables, chart-inspired decorative elements, confidence interval / uncertainty visualization aesthetics |
| AI/ML | Node-and-edge decorative patterns (subtle), smooth animations, modern geometric shapes, code-integrated design |

---

## Design Preferences (Detailed)

### Overall Aesthetic
- **Clean, modern, academic** — Think Nature journal meets Stripe's technical docs
- **Professional but approachable** — Credible for academic peers, welcoming for students
- **Data-informed design** — Let visualizations and code be first-class visual elements
- **Generous whitespace** — Allow content to breathe; avoid clutter
- **Confident minimalism** — Every element should earn its place
- **Scientific credibility** — Avoid gimmicks; prioritize clarity and precision

### Tone & Personality
- **Authoritative but not intimidating** — Expert who explains clearly
- **Curious and exploratory** — Invites readers to learn alongside
- **Precise language** — Reflects statistical and scientific training
- **Modern and forward-thinking** — AI/ML focus feels cutting-edge
- **Grounded in real-world applications** — Climate science has purpose and impact

---

## Color System

### Primary Palette

#### Light Mode
| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| **Primary** | Deep Ocean Teal | `#0D7377` | Headers, links, primary buttons, accents |
| **Primary Dark** | Deep Sea | `#065A5A` | Hover states, emphasized elements |
| **Primary Light** | Soft Teal | `#14A3A8` | Highlights, secondary accents |
| **Secondary** | Warm Coral | `#E07A5F` | Call-to-action buttons, important highlights, accent |
| **Secondary Light** | Soft Coral | `#F4A582` | Hover states, subtle accents |
| **Background** | Clean White | `#FAFBFC` | Page background |
| **Surface** | Light Gray | `#F1F5F9` | Cards, code block backgrounds in light mode prose |
| **Text Primary** | Charcoal | `#1E293B` | Body text |
| **Text Secondary** | Slate | `#64748B` | Captions, metadata, secondary text |
| **Border** | Light Border | `#E2E8F0` | Dividers, card borders |

#### Dark Mode
| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| **Primary** | Bright Teal | `#2DD4BF` | Headers, links, primary buttons |
| **Primary Muted** | Soft Teal | `#5EEAD4` | Highlights |
| **Secondary** | Warm Coral | `#FB923C` | Accents, CTAs |
| **Background** | Deep Navy | `#0F172A` | Page background |
| **Surface** | Dark Slate | `#1E293B` | Cards, elevated surfaces |
| **Surface Elevated** | Slate | `#334155` | Hover states, modals |
| **Text Primary** | Off White | `#F1F5F9` | Body text |
| **Text Secondary** | Cool Gray | `#94A3B8` | Captions, metadata |
| **Border** | Dark Border | `#334155` | Dividers, card borders |

### Semantic Colors
| Purpose | Light Mode | Dark Mode |
|---------|------------|-----------|
| Success | `#10B981` | `#34D399` |
| Warning | `#F59E0B` | `#FBBF24` |
| Error | `#EF4444` | `#F87171` |
| Info | `#3B82F6` | `#60A5FA` |

### Gradient Accents (Use Sparingly)
- **Hero gradient**: `linear-gradient(135deg, #0D7377 0%, #14A3A8 50%, #2DD4BF 100%)` — Evokes ocean/atmosphere depth
- **Temperature gradient** (for data viz): `#3B82F6 → #10B981 → #FBBF24 → #EF4444` — Cool to warm, scientific
- **Subtle background texture**: Very faint topographic contour lines or grid pattern at 3-5% opacity

### Code Block Colors
**Always use dark theme for code blocks, even in light mode:**
- Background: `#1E1E1E` (VS Code dark)
- Text: Standard syntax highlighting (VS Code Dark+ or similar)
- This maintains readability and feels familiar to developers

---

## Typography System

### Font Stack

| Role | Font Family | Fallback | Weight |
|------|-------------|----------|--------|
| **Headings** | Inter | system-ui, sans-serif | 600-700 |
| **Body** | Inter or Source Sans 3 | system-ui, sans-serif | 400-500 |
| **Code** | JetBrains Mono or Fira Code | monospace | 400-500 |
| **Math** | KaTeX default | serif | — |

> **Alternative**: Use "Plus Jakarta Sans" for headings for a slightly warmer academic feel

### Type Scale

| Element | Size (Desktop) | Size (Mobile) | Line Height | Weight |
|---------|----------------|---------------|-------------|--------|
| H1 (Page title) | 3rem (48px) | 2.25rem (36px) | 1.1 | 700 |
| H2 (Section) | 2.25rem (36px) | 1.75rem (28px) | 1.2 | 600 |
| H3 (Subsection) | 1.5rem (24px) | 1.25rem (20px) | 1.3 | 600 |
| H4 | 1.25rem (20px) | 1.125rem (18px) | 1.4 | 600 |
| Body | 1.125rem (18px) | 1rem (16px) | 1.7 | 400 |
| Body Small | 0.875rem (14px) | 0.875rem | 1.6 | 400 |
| Code | 0.9rem (14.4px) | 0.85rem | 1.5 | 400 |
| Caption | 0.75rem (12px) | 0.75rem | 1.5 | 500 |

### Typography Rules
- **Max prose width**: 65-75 characters (~700-750px) for optimal readability
- **Paragraph spacing**: 1.5em between paragraphs
- **Heading spacing**: 2em above, 0.75em below
- **Letter spacing**: Slight negative for large headings (-0.02em), normal for body
- **Links**: Primary color, underline on hover, subtle transition

---

## Layout System

### Grid & Spacing

| Token | Value | Usage |
|-------|-------|-------|
| `--space-1` | 4px | Tight spacing |
| `--space-2` | 8px | Related elements |
| `--space-3` | 12px | Default gap |
| `--space-4` | 16px | Card padding |
| `--space-6` | 24px | Section spacing |
| `--space-8` | 32px | Major sections |
| `--space-12` | 48px | Page sections |
| `--space-16` | 64px | Hero spacing |
| `--space-24` | 96px | Page margins |

### Container Widths

| Container | Max Width | Usage |
|-----------|-----------|-------|
| Narrow | 680px | Blog prose, About text |
| Default | 1024px | Most content pages |
| Wide | 1280px | Code + visualization layouts |
| Full | 100% | Hero sections, full-bleed |

### Page Layout Patterns

#### Standard Page
```
┌─────────────────────────────────────────────────────┐
│  Navigation (sticky)                                │
├─────────────────────────────────────────────────────┤
│                                                     │
│    ┌───────────────────────────────────────────┐    │
│    │  Page Title (H1)                          │    │
│    │  Subtitle / Metadata                      │    │
│    └───────────────────────────────────────────┘    │
│                                                     │
│    ┌───────────────────────────────────────────┐    │
│    │                                           │    │
│    │  Main Content                             │    │
│    │  (max-width: narrow or default)           │    │
│    │                                           │    │
│    └───────────────────────────────────────────┘    │
│                                                     │
├─────────────────────────────────────────────────────┤
│  Footer                                             │
└─────────────────────────────────────────────────────┘
```

#### Blog Chunk Page (Code + Theory)
```
Desktop (≥1024px):
┌─────────────────────────────────────────────────────────────────┐
│  Navigation                                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┬────────────────────────────────────────────────┐   │
│  │         │                                                │   │
│  │  TOC    │   Theory Section (Markdown + Math)             │   │
│  │  (fixed)│   - Full width prose                           │   │
│  │         │                                                │   │
│  │         ├────────────────────────────────────────────────┤   │
│  │         │                                                │   │
│  │         │   ┌──────────────────┬─────────────────────┐   │   │
│  │         │   │  Code Editor     │  Output Panel       │   │   │
│  │         │   │  (Monaco)        │  - Console          │   │   │
│  │         │   │                  │  - Visualization    │   │   │
│  │         │   │  [Run][Reset]    │                     │   │   │
│  │         │   └──────────────────┴─────────────────────┘   │   │
│  │         │                                                │   │
│  │         │   More Theory...                               │   │
│  │         │                                                │   │
│  └─────────┴────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Chunk Navigation: ← Previous  |  1 2 3 4 5 6  |  Next → │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Mobile (<1024px):
┌─────────────────────────────┐
│  Navigation (hamburger)     │
├─────────────────────────────┤
│  Theory Section             │
│  (full width, stacked)      │
├─────────────────────────────┤
│  Code Editor (full width)   │
│  [Run] [Reset] [Copy]       │
├─────────────────────────────┤
│  Output (collapsible)       │
├─────────────────────────────┤
│  More Theory...             │
├─────────────────────────────┤
│  ← Prev  [Progress]  Next → │
└─────────────────────────────┘
```

---

## Component Design Specifications

### Navigation Bar
- **Position**: Fixed/sticky at top
- **Height**: 64px desktop, 56px mobile
- **Background**: Semi-transparent with backdrop blur (`rgba(255,255,255,0.8)` + blur)
- **Elements**: Logo/Name (left), Nav links (center or right), Theme toggle (right)
- **Mobile**: Hamburger menu with slide-out drawer
- **Active state**: Underline or highlight on current page
- **Shadow**: Subtle shadow on scroll (`box-shadow: 0 1px 3px rgba(0,0,0,0.1)`)

### Cards (Projects, Publications, Blog Series)
- **Background**: Surface color
- **Border**: 1px solid border color, OR no border with subtle shadow
- **Border radius**: 12px
- **Padding**: 24px
- **Hover**: Slight lift (`transform: translateY(-2px)`) + shadow increase
- **Transition**: 200ms ease-out
- **Image**: Top of card, aspect ratio 16:9 or 3:2, with subtle zoom on hover

### Buttons

| Variant | Style |
|---------|-------|
| **Primary** | Solid primary color, white text, rounded-lg (8px), padding 12px 24px |
| **Secondary** | Outline with primary color border, primary text, transparent bg |
| **Ghost** | No border, primary text, subtle hover background |
| **Danger** | Error color background (admin only) |

- **Hover**: Darken 10% or add shadow
- **Active**: Darken 15%, slight scale down (0.98)
- **Disabled**: 50% opacity, no pointer events
- **Icon buttons**: Equal padding all sides, centered icon

### Code Editor Component
- **Container**: Dark background (`#1E1E1E`), rounded-lg, subtle border
- **Header bar**: Filename display, language badge, action buttons (Run, Reset, Copy, Download)
- **Editor area**: Monaco Editor with line numbers, proper indentation guides
- **Minimum height**: 200px, resizable vertically
- **Max height**: 500px with scroll
- **Action buttons**: Icon + text on desktop, icon-only on mobile

### Output Panel
- **Split view**: 50/50 with code editor on desktop, stacked on mobile
- **Tabs**: "Console" | "Visualization" (if applicable)
- **Console output**: Monospace font, scrollable, dark background
- **Visualization**: Centered image, max-width 100%, click to expand
- **Loading state**: Spinner + "Running code..." message
- **Error display**: Red-tinted background, clear error message, line number if available

### Math Rendering (KaTeX)
- **Inline math**: Flows with text naturally
- **Display math**: Centered, with generous vertical margin (24px above/below)
- **Equation numbers**: Right-aligned (if used)
- **Overflow**: Horizontal scroll for wide equations on mobile

### Tags & Badges
- **Style**: Pill-shaped (full border-radius), small padding (4px 12px)
- **Colors**: Category-specific (e.g., Python = blue, Climate = teal, ML = purple)
- **Size**: Small text (12-14px), uppercase or sentence case
- **Grouping**: Flex wrap with small gap (8px)

### Timeline Component (Education/Experience)
- **Layout**: Vertical line with nodes
- **Nodes**: Circle markers on the line, cards extending to one side
- **Animation**: Fade-in on scroll (optional)
- **Dates**: Clear year markers
- **Responsive**: Stack naturally on mobile

### Progress Indicator (Blog Series)
- **Style**: Horizontal dots or numbered steps
- **Current**: Filled/highlighted
- **Completed**: Filled with checkmark or different color
- **Upcoming**: Outline only
- **Clickable**: Navigate to any chunk

### Table of Contents (Blog Posts)
- **Desktop**: Fixed sidebar, left side, scrolls with content highlighting
- **Mobile**: Collapsible dropdown at top of content
- **Active section**: Highlighted as user scrolls
- **Smooth scroll**: On click

---

## Admin Dashboard Design

### Design Principles for Admin
- **Functional over decorative** — Prioritize usability
- **Consistent with main site** — Same color palette, fonts, but simpler
- **Dense but organized** — More information per screen than public site
- **Clear hierarchy** — Easy to find actions

### Admin Layout
```
┌─────────────────────────────────────────────────────────────────┐
│  Admin Header: Logo | "Admin Dashboard" | User | Logout         │
├──────────────┬──────────────────────────────────────────────────┤
│              │                                                  │
│   Sidebar    │   Main Content Area                              │
│   ────────   │                                                  │
│   Dashboard  │   ┌──────────────────────────────────────────┐   │
│   Blog Posts │   │  Page Title                              │   │
│   ├─ Series  │   │  ─────────────────────────────────────   │   │
│   ├─ Chunks  │   │                                          │   │
│   Publications│  │  Content / Forms / Tables                │   │
│   Projects   │   │                                          │   │
│   Pages      │   │                                          │   │
│   ├─ About   │   │                                          │   │
│   ├─ Contact │   │                                          │   │
│   Media      │   │                                          │   │
│   Settings   │   └──────────────────────────────────────────┘   │
│              │                                                  │
└──────────────┴──────────────────────────────────────────────────┘
```

### Admin Components

#### Data Tables
- **Columns**: Selectable/sortable where appropriate
- **Actions**: Edit, Delete, View buttons per row
- **Bulk actions**: Checkbox selection + action dropdown
- **Pagination**: Bottom of table, show count
- **Search/Filter**: Top of table
- **Empty state**: Helpful message + CTA to add first item

#### Forms
- **Layout**: Single column, grouped sections with headers
- **Labels**: Above inputs, clear and concise
- **Required fields**: Marked with asterisk or "Required" text
- **Validation**: Inline error messages below fields
- **Save actions**: Sticky bottom bar with "Save Draft" and "Publish" buttons

#### Markdown Editor (for theory content)
- **Split view**: Editor on left, live preview on right
- **Toolbar**: Bold, Italic, Headers, Lists, Links, Images, Code blocks, Math insertion
- **Image upload**: Drag-and-drop zone or button
- **Math preview**: Renders KaTeX in preview pane
- **Full-screen mode**: Option to expand editor

#### Code Editor (for code.py files)
- **Monaco Editor**: Same as public-facing but full-featured
- **Syntax validation**: Show Python errors
- **Test run**: Button to execute and verify code works
- **Template snippets**: Quick insert for common patterns

#### Media Library
- **Grid view**: Thumbnails with filename
- **Upload**: Drag-and-drop zone
- **Organization**: Folders or tags
- **Usage tracking**: Show where each image is used
- **Delete confirmation**: Warn if image is in use

### Admin Blog Creation Flow

#### Adding a New Blog Series
1. Click "New Series" button
2. Fill form:
   - Title
   - Slug (auto-generated, editable)
   - Description
   - Paper metadata (authors, year, arxiv link, venue)
   - Difficulty level (dropdown: Beginner, Intermediate, Advanced)
   - Prerequisites (tag input)
   - Cover image upload
   - Status (Draft / Published)
3. Save → Redirects to series page where chunks can be added

#### Adding a New Chunk
1. From series page, click "Add Chunk"
2. Fill form:
   - Chunk number (auto-incremented, editable)
   - Title
   - Estimated reading time
   - Theory content (Markdown editor with preview)
   - Code blocks (can add multiple):
     - Code title/description
     - Python code (Monaco editor)
     - Expected output type (console/visualization/both)
   - Figures (upload + caption + alt text)
   - Status (Draft / Published)
3. Save → Can preview chunk as it would appear on site
4. Publish → Goes live

#### Blog Management Table Columns
| Column | Description |
|--------|-------------|
| Title | Series/Chunk name (clickable to edit) |
| Status | Draft / Published badge |
| Chunks | Count of chunks (for series) |
| Last Modified | Date/time |
| Views | Page view count (if analytics enabled) |
| Actions | Edit, Preview, Delete |

---

## Animation & Interaction Guidelines

### Principles
- **Purposeful motion** — Animations should guide, not distract
- **Subtle and fast** — Typically 150-300ms
- **Consistent easing** — Use `ease-out` for entrances, `ease-in` for exits

### Specific Animations

| Element | Animation | Duration | Easing |
|---------|-----------|----------|--------|
| Page transitions | Fade in | 200ms | ease-out |
| Card hover | Lift + shadow | 200ms | ease-out |
| Button hover | Background shift | 150ms | ease |
| Menu open | Slide down/right | 250ms | ease-out |
| Modal open | Fade + scale up (0.95→1) | 200ms | ease-out |
| Code output appear | Fade in | 300ms | ease-out |
| TOC highlight | Background slide | 200ms | ease |
| Scroll to section | Smooth scroll | 500ms | ease-in-out |

### Loading States
- **Page load**: Skeleton screens for cards/content areas
- **Code execution**: Spinner + "Running..." text, disable Run button
- **Form submit**: Button shows spinner, disabled state
- **Image loading**: Blur placeholder → sharp image (if using next/image)

### Micro-interactions
- Button press: Slight scale down (0.98)
- Copy button: Changes to "Copied!" with checkmark for 2s
- Like/Save: Heart fill animation (if implementing)
- Toast notifications: Slide in from top-right, auto-dismiss after 5s

---

## Iconography

### Icon Library
**Recommended**: Lucide Icons (consistent, clean, MIT license)
- Matches modern design aesthetic
- Good variety for tech/academic use
- React component available

### Icon Usage

| Context | Icons to Use |
|---------|--------------|
| Navigation | Home, User, BookOpen, Briefcase, FileText, Mail |
| Actions | Play, RotateCcw, Copy, Download, ExternalLink, Edit, Trash |
| Status | Check, X, AlertCircle, Info, Loader |
| Social | Github, Linkedin, Twitter, Mail, Globe |
| Content | Code, FileCode, Image, Folder, Tag |
| Theme | Sun, Moon |

### Icon Sizing
| Context | Size |
|---------|------|
| Inline with text | 16-18px (1em) |
| Buttons | 18-20px |
| Navigation | 20-24px |
| Feature highlights | 24-32px |
| Empty states | 48-64px |

---

## Data Visualization Styling

> As a climate scientist and statistician, data visualizations should feel professional and publication-ready.

### Chart Defaults (for any embedded visualizations)
- **Font**: Same as site (Inter or system)
- **Colors**: Use color palette (primary, secondary, semantic)
- **Grid**: Light gray, subtle (#E2E8F0 in light mode)
- **Axis labels**: Clear, with units
- **Legends**: Outside plot area when possible
- **Responsive**: Scale appropriately

### Matplotlib Styling (for Pyodide outputs)
Include a default style in code templates:
```python
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
```

### Climate/Science-Specific
- Temperature colormaps: Use perceptually uniform (viridis, plasma) or diverging (coolwarm, RdBu)
- Uncertainty: Show with shaded bands or error bars
- Time series: Clear x-axis date formatting
- Maps (if any): Consider colorblind-friendly palettes

---

## Responsive Design Breakpoints

| Breakpoint | Width | Target |
|------------|-------|--------|
| `sm` | 640px | Large phones |
| `md` | 768px | Tablets |
| `lg` | 1024px | Small laptops |
| `xl` | 1280px | Desktops |
| `2xl` | 1536px | Large screens |

### Mobile-First Priorities
1. Readable text without zooming
2. Tappable buttons (min 44x44px touch targets)
3. No horizontal scroll
4. Collapsible/accordion sections for dense content
5. Bottom navigation for key actions (if app-like)

---

## Accessibility Requirements

### WCAG 2.1 AA Compliance
- **Color contrast**: Minimum 4.5:1 for text, 3:1 for large text
- **Focus indicators**: Visible focus rings on interactive elements
- **Alt text**: All images must have descriptive alt text
- **Keyboard navigation**: All interactive elements reachable via Tab
- **ARIA labels**: For icon-only buttons and complex widgets
- **Skip links**: "Skip to main content" for screen readers
- **Reduced motion**: Respect `prefers-reduced-motion` media query

### Math Accessibility
- KaTeX renders as readable by screen readers
- Provide text descriptions for complex equations when possible

---

## Performance Guidelines

### Targets
- **Lighthouse score**: 90+ on all metrics
- **First Contentful Paint**: < 1.5s
- **Largest Contentful Paint**: < 2.5s
- **Cumulative Layout Shift**: < 0.1
- **Time to Interactive**: < 3.5s

### Optimization Strategies
- **Images**: WebP format, responsive srcset, lazy loading
- **Fonts**: Subset fonts, preload critical fonts, font-display: swap
- **Code splitting**: Lazy load Monaco Editor and Pyodide
- **Static generation**: Pre-render all possible pages
- **Bundle size**: Tree-shake unused components, analyze bundle regularly
- **Caching**: Aggressive caching for static assets

### Pyodide Loading Strategy
1. Don't load on initial page visit
2. Show "Loading Python environment..." when user first interacts with code
3. Cache Pyodide in IndexedDB for return visits
4. Consider loading during idle time after main content

---

## SEO Requirements

### Meta Tags (per page)
- Title: `{Page Title} | Shammunul Islam`
- Description: Unique, 150-160 characters
- Open Graph: title, description, image
- Twitter Card: summary_large_image

### Structured Data
- Person schema (for About)
- Article schema (for blog posts)
- BreadcrumbList schema

### Technical SEO
- Sitemap.xml (auto-generated)
- Robots.txt (allow all, block /admin)
- Canonical URLs
- Clean URL structure (no trailing slashes)

---

## Tech Stack Suggestions

> Claude Code can suggest alternatives, but here are preferences:

### Frontend Framework
**Preferred: Next.js 14+ (App Router)**
- React-based (good ecosystem)
- SSG for static pages
- API routes for admin/backend
- Good Vercel deployment

**Alternative: Astro**
- If mostly static with islands of interactivity
- Lighter weight

### Styling
**Preferred: Tailwind CSS**
- Utility-first
- Easy dark mode
- Good with component libraries

### UI Components
- shadcn/ui (works great with Tailwind)
- Or Radix UI primitives

### Code Editor
**Preferred: Monaco Editor**
- Same editor as VS Code
- Excellent Python support
- Rich features

**Alternative: CodeMirror 6**
- Lighter weight
- Also excellent

### Python in Browser
**Required: Pyodide**
- CPython compiled to WebAssembly
- Supports numpy, scipy, matplotlib
- https://pyodide.org

### Markdown Processing
- MDX (Markdown + JSX components)
- Math: KaTeX (faster) or MathJax
- Syntax highlighting: Shiki or Prism

### Database
**For admin/comments, choose one:**
- Supabase (PostgreSQL, easy auth, free tier)
- Firebase (NoSQL, good free tier)
- PlanetScale (MySQL, serverless)
- Or: File-based/Git-based for simplicity (no database)

### Authentication (Admin)
- NextAuth.js
- Or Supabase Auth
- Or simple password protection

### Deployment
**Preferred: Vercel**
- Seamless Next.js deployment
- Free tier sufficient
- Good analytics

**Alternative: Netlify**

### Other Tools
- Image optimization: next/image or sharp
- Analytics: Vercel Analytics or Plausible
- Forms: Formspree or built-in API route

---

## Current Content Available

### FNO Paper Series (6 Chunks)
**Location in project:** `/blogs/ai-paper-implementations/fno-paper/`

| Chunk | Topic | Files |
|-------|-------|-------|
| 1 | Mathematical Foundations | theory.md, code.py |
| 2 | Fourier Theory for Neural Networks | theory.md, code.py |
| 3 | Neural Operator Framework | theory.md, code.py |
| 4 | FNO Architecture | theory.md, code.py |
| 5 | Training Methodology | theory.md, code.py |
| 6 | Applications & Benchmarks | theory.md, code.py |

**Total**: 6 theory files + 6 code files + figures

### Content Format Notes
- Theory files: Markdown with LaTeX math ($ and $$ delimiters)
- Code files: Python 3, use numpy/scipy/matplotlib
- Code includes print statements and visualization generation
- Each code file is self-contained and runnable

---

## File Structure

```
portfolio-website/
│
├── CLAUDE_CONTEXT.md              # THIS FILE - project context
├── README.md                      # Project readme for GitHub
├── package.json                   # Dependencies (to be created)
│
├── personal/                      # Owner's personal information
│   ├── bio.md                     # About me content
│   ├── cv.pdf                     # Downloadable CV
│   ├── links.json                 # Social/professional links
│   ├── publications.json          # Papers, books list
│   └── headshot.jpg               # Profile photo
│
├── blogs/                         # All blog content
│   └── ai-paper-implementations/  # Main blog section
│       ├── section_metadata.json  # Section info
│       │
│       └── fno-paper/             # First paper series
│           ├── metadata.json      # Paper metadata
│           │
│           └── chunks/
│               ├── chunk0/
│               │   └── introduction.md
│               │
│               ├── chunk1/
│               │   ├── chunk1_complete.md   # Merged theory + code
│               │   ├── theory.md            # (optional) standalone theory
│               │   ├── code.py              # (optional) standalone code
│               │   └── figures/
│               │       ├── 01_function_vs_operator.png
│               │       ├── 02_discretization_problem.png
│               │       ├── 03_sinusoid_components.png
│               │       ├── 04_dft_implementation.png
│               │       ├── 05_frequency_meaning.png
│               │       ├── 06_2d_fft.png
│               │       ├── 07_differentiation_property.png
│               │       ├── 08_convolution_theorem.png
│               │       └── 09_pde_spectral_solution.png
│               │
│               ├── chunk2/
│               │   ├── chunk2_complete.md
│               │   └── figures/
│               │       └── *.png
│               │
│               ├── chunk3/
│               │   ├── chunk3_complete.md
│               │   └── figures/
│               │       └── *.png
│               │
│               ├── chunk4/
│               │   ├── chunk4_complete.md
│               │   └── figures/
│               │       └── *.png
│               │
│               ├── chunk5/
│               │   ├── chunk5_complete.md
│               │   └── figures/
│               │       └── *.png
│               │
│               └── chunk6/
│                   ├── chunk6_complete.md
│                   └── figures/
│                       └── *.png
│
├── projects/                      # Projects showcase data
│   └── projects.json
│
├── config/                        # Site configuration
│   └── site_config.json
│
├── src/                           # Source code (to be created)
│   ├── app/                       # Next.js app router pages
│   ├── components/                # React components
│   ├── lib/                       # Utilities
│   └── styles/                    # Global styles
│
└── public/                        # Static assets
    └── images/
```

---

## Priority Order for Development

### Phase 1: Foundation
1. Initialize Next.js project with Tailwind
2. Set up basic layout (header, footer, nav)
3. Create Home page (static content first)
4. Create About page
5. Create Contact page with form
6. Basic dark/light mode toggle

### Phase 2: Content Pages
7. Research & Publications page
8. Projects page
9. CV page with PDF download
10. Import personal content (bio, publications, projects data)

### Phase 3: Blog Infrastructure
11. "AI Paper Implementations" landing page
12. Blog series listing component
13. Individual series page (FNO paper)
14. Chunk page with markdown rendering
15. Code syntax highlighting (read-only)
16. Math rendering (KaTeX)
17. Navigation between chunks

### Phase 4: Interactivity
18. Integrate Monaco Editor
19. Set up Pyodide for Python execution
20. Code run/reset/copy/download buttons
21. Output display (console + visualizations)
22. Real-time visualization updates
23. Handle errors gracefully
24. Loading states and UX polish

### Phase 5: Admin Dashboard
25. Admin authentication
26. Admin layout and navigation
27. Blog series CRUD
28. Chunk CRUD with editors
29. Image upload
30. Draft/publish system
31. Site content editing

### Phase 6: Polish & Launch
32. SEO optimization (meta tags, sitemap)
33. Performance optimization
34. Mobile responsiveness testing
35. Accessibility review
36. Analytics integration
37. Final testing
38. Deploy to production

---

## Build Progress

> Update this section as development progresses

| Section/Feature | Status | Notes | Date |
|-----------------|--------|-------|------|
| Project Setup | ⬜ Not started | | |
| Home | ⬜ Not started | | |
| About | ⬜ Not started | | |
| AI Paper Implementations | ⬜ Not started | FNO content ready (6 chunks) | |
| Research & Publications | ⬜ Not started | | |
| Projects | ⬜ Not started | | |
| CV | ⬜ Not started | | |
| Contact | ⬜ Not started | | |
| Code Editor | ⬜ Not started | | |
| Pyodide Integration | ⬜ Not started | | |
| Admin Dashboard | ⬜ Not started | | |
| Dark/Light Mode | ⬜ Not started | | |
| Mobile Responsive | ⬜ Not started | | |
| Deployment | ⬜ Not started | | |

**Legend:** ⬜ Not started | 🔄 In progress | ✅ Done | ❌ Blocked

---

## Questions for Claude Code

When starting implementation, consider:

1. What's the best project structure for Next.js App Router?
2. How to efficiently load Pyodide (it's large ~10MB)?
3. Best pattern for the admin dashboard?
4. How to handle markdown + code file relationships?
5. Should content be in the repo or database?
6. How to make code execution secure?
7. Best approach for the chunk navigation UX?

---

## Additional Notes

- This is a personal portfolio site, not high-traffic commercial
- Single maintainer (site owner)
- Content will grow over time (more papers, more series)
- Prioritize maintainability over complexity
- Open to implementation suggestions
- Can iterate and improve after initial launch

---

## Contact for Clarifications

If any requirements are unclear during development, the priority is:
1. Check this document
2. Check existing content files for patterns
3. Make reasonable assumptions and document them
4. Ask for clarification in commit messages/comments

---

*Last updated: December 2025*
*Document version: 2.0*
