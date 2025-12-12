# Visual Design Style Guide
## "Clean Educational" Style System

This document describes the design system used for creating clean, professional educational diagrams and illustrations.

---

## 🎨 COLOR PALETTE

### Background
- **Primary Background**: `#fafbff` → `#f5f7ff` (subtle blue-tinted white gradient)
- Creates a soft, non-distracting canvas

### Accent Colors (Semantic)

#### Warm Colors (Input/Before/Problem)
| Purpose | Color | Hex |
|---------|-------|-----|
| Lightest | Cream | `#fef3c7` |
| Light | Yellow | `#fcd34d` |
| Medium | Orange | `#f97316` |
| Dark | Red-Orange | `#dc2626` |
| Text/Border | Burnt Orange | `#c2410c`, `#ea580c` |
| Badge BG | Light Orange | `#fff7ed` |
| Border | Peach | `#fdba74` |

#### Cool Colors (Output/After/Solution)
| Purpose | Color | Hex |
|---------|-------|-----|
| Lightest | Mint | `#ecfdf5` |
| Light | Light Teal | `#6ee7b7` |
| Medium | Teal | `#14b8a6` |
| Dark | Cyan | `#0891b2` |
| Text | Deep Teal | `#0d9488`, `#047857` |
| Border | Soft Green | `#6ee7b7` |

#### Purple/Violet (Operators/Actions/Transformations)
| Purpose | Color | Hex |
|---------|-------|-----|
| Light | Lavender | `#c4b5fd` |
| Medium | Violet | `#a78bfa` |
| Dark | Purple | `#8b5cf6`, `#7c3aed` |

#### Neutrals
| Purpose | Color | Hex |
|---------|-------|-----|
| Heading Text | Dark Slate | `#1e293b` |
| Body Text | Gray | `#64748b` |
| Muted Text | Light Gray | `#94a3b8` |
| Borders | Very Light | `#e2e8f0` |
| White | Pure | `#ffffff` |

---

## 📝 TYPOGRAPHY

### Font Stack
```
font-family: system-ui, -apple-system, sans-serif;
```
For mathematical notation:
```
font-family: Georgia, serif;
```

### Text Hierarchy

| Element | Size | Weight | Color | Style |
|---------|------|--------|-------|-------|
| Main Title | 28px | 700 (bold) | `#1e293b` | — |
| Subtitle | 14px | 400 | `#64748b` | — |
| Section Header | 13px | 600 | `#64748b` | uppercase, letter-spacing: 0.5 |
| Math Symbols | 28-42px | 600 | accent color | italic (Georgia) |
| Labels | 11px | 500 | accent color | — |
| Small Text | 10px | 400 | `#94a3b8` | — |

### Text Styling
- **Headings**: Clean, no decorations
- **Labels**: Sentence case or Title Case
- **Math**: Georgia serif, italic for variables
- **Uppercase**: Only for category labels (INPUT FUNCTION, OUTPUT FUNCTION)

---

## 📦 LAYOUT & SPACING

### Card Design
```svg
<rect rx="16" fill="white" filter="url(#shadow)"/>
```
- **Border Radius**: 16px for cards, 8px for inner elements, 13px for badges
- **Card Background**: Pure white (`#ffffff`)
- **Padding**: ~30px internal padding

### Spacing Principles
- **Generous whitespace** between elements
- **Consistent margins** (30-40px between major sections)
- **Vertical rhythm**: Keep consistent spacing between text lines

### Grid Heatmaps
- Cell size: 44×28px
- Border radius: 2px per cell
- Container border radius: 8px
- Container stroke: 2px, accent color

---

## 🌟 EFFECTS & FILTERS

### Primary Shadow (Cards)
```svg
<filter id="shadow">
  <feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#6366f1" flood-opacity="0.12"/>
</filter>
```
- Purple-tinted shadow
- Soft, not harsh
- Creates subtle depth

### Small Shadow (Inner elements)
```svg
<filter id="smallShadow">
  <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#000" flood-opacity="0.08"/>
</filter>
```

### Key Principles
- **No harsh shadows** - always soft and subtle
- **Purple-tinted shadows** for a modern feel
- **Low opacity** (0.08-0.12)
- **Vertical offset only** (dx="0")

---

## 🔷 SHAPES & ELEMENTS

### Arrows/Operators
```svg
<!-- Simple directional arrow -->
<path d="M0,85 L220,85 L220,50 L300,100 L220,150 L220,115 L0,115 Z" fill="url(#operatorGrad)"/>
```
- Clean geometric shapes
- Gradient fills (light → dark in direction of flow)
- Rounded internal elements (circles for symbols)

### Badges/Pills
```svg
<rect rx="13" fill="#fff7ed"/>  <!-- height ~26px -->
<rect rx="25" fill="white"/>    <!-- height ~50px -->
```
- Border radius = half of height for pill shape
- Light tinted backgrounds matching section color
- Optional: subtle border in accent color

### Flow Indicators
```svg
<g fill="#c4b5fd" opacity="0.4">
  <circle cx="365" cy="265" r="4"/>
  <circle cx="380" cy="265" r="4"/>
  <circle cx="395" cy="265" r="4"/>
</g>
```
- Three dots in a row
- Low opacity (0.4)
- Accent color matching the section

### Color Scale Legends
```svg
<text>Low</text>
<rect width="80" height="12" rx="2" fill="url(#gradient)"/>
<text>High</text>
```
- Simple horizontal bar
- "Low" and "High" labels
- Small size (10px text, 12px bar height)

---

## 📐 GRADIENTS

### Background Gradient
```svg
<linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" style="stop-color:#fafbff"/>
  <stop offset="100%" style="stop-color:#f5f7ff"/>
</linearGradient>
```

### Heatmap Gradients
```svg
<!-- Warm: diagonal, bottom-left to top-right -->
<linearGradient id="warmGrad" x1="0%" y1="100%" x2="100%" y2="0%">
  <stop offset="0%" style="stop-color:#fef3c7"/>
  <stop offset="35%" style="stop-color:#fcd34d"/>
  <stop offset="65%" style="stop-color:#f97316"/>
  <stop offset="100%" style="stop-color:#dc2626"/>
</linearGradient>

<!-- Cool: same direction -->
<linearGradient id="coolGrad" x1="0%" y1="100%" x2="100%" y2="0%">
  <stop offset="0%" style="stop-color:#ecfdf5"/>
  <stop offset="35%" style="stop-color:#6ee7b7"/>
  <stop offset="65%" style="stop-color:#14b8a6"/>
  <stop offset="100%" style="stop-color:#0891b2"/>
</linearGradient>
```

### Operator Gradient
```svg
<!-- Horizontal, light to dark -->
<linearGradient id="operatorGrad" x1="0%" y1="0%" x2="100%" y2="0%">
  <stop offset="0%" style="stop-color:#c4b5fd"/>
  <stop offset="50%" style="stop-color:#a78bfa"/>
  <stop offset="100%" style="stop-color:#8b5cf6"/>
</linearGradient>
```

---

## ✅ DESIGN PRINCIPLES

### 1. **Minimalism**
- Remove unnecessary elements
- Every element should serve a purpose
- Prefer whitespace over decoration

### 2. **Clarity**
- Information hierarchy through size and weight
- Clear visual flow (left → right, top → bottom)
- Semantic colors (warm=input, cool=output, purple=action)

### 3. **Consistency**
- Same border radius across similar elements
- Consistent shadow treatment
- Uniform spacing

### 4. **Softness**
- Rounded corners everywhere
- Soft shadows with low opacity
- Gentle gradients
- No harsh contrasts

### 5. **Professional Academic Feel**
- Clean sans-serif for text
- Elegant serif for math
- Muted, sophisticated color palette
- No decorative elements or emojis in main content

---

## 📋 QUICK REFERENCE TEMPLATE

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 600">
  <defs>
    <!-- Background -->
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#fafbff"/>
      <stop offset="100%" style="stop-color:#f5f7ff"/>
    </linearGradient>
    
    <!-- Shadow -->
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">
      <feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#6366f1" flood-opacity="0.12"/>
    </filter>
  </defs>
  
  <!-- Background -->
  <rect width="1100" height="600" fill="url(#bgGrad)"/>
  
  <!-- Title -->
  <text x="550" y="45" text-anchor="middle" 
        font-family="system-ui, -apple-system, sans-serif" 
        font-size="28" font-weight="700" fill="#1e293b">
    Title Here
  </text>
  
  <!-- Card -->
  <rect x="70" y="100" width="280" height="300" rx="16" 
        fill="white" filter="url(#shadow)"/>
</svg>
```

---

## 🏷️ STYLE NAME

When requesting future diagrams, reference this style as:

**"Clean Educational Style"** or **"Shammun's Preferred Style"**

Key characteristics to mention:
- Soft blue-tinted background
- White cards with purple-tinted shadows
- Warm orange/cool teal semantic colors
- Purple for operators/actions
- System-ui font with Georgia for math
- Minimalist with generous whitespace
- Grid heatmaps for data visualization
- Pill-shaped badges and labels
