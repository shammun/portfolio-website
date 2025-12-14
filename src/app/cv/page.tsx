import Link from "next/link";
import type { Metadata } from "next";
import {
  Download,
  ExternalLink,
  GraduationCap,
  Briefcase,
  Award,
  BookOpen,
} from "lucide-react";

export const metadata: Metadata = {
  title: "CV",
  description:
    "Curriculum Vitae of Shammunul Islam - Climate Scientist and ML Researcher",
};

const highlights = [
  {
    icon: GraduationCap,
    title: "Education",
    color: "cool",
    items: [
      "MS Climate Science - George Mason University (2025)",
      "MA Climate & Society - Columbia University (2014)",
      "BSc Statistics - SUST, Bangladesh",
    ],
  },
  {
    icon: Briefcase,
    title: "Current Roles",
    color: "warm",
    items: [
      "Local Climate Action Fellow - Virginia Climate Center",
      "MS Thesis: Urban Heat Island Prediction using FNO",
    ],
  },
  {
    icon: BookOpen,
    title: "Publications",
    color: "primary",
    items: [
      "2 Books on GIS & Remote Sensing (Packt Publishing)",
      "3 Peer-reviewed Journal Articles",
      "AGU25 Conference Presentation",
    ],
  },
  {
    icon: Award,
    title: "Expertise",
    color: "violet",
    items: [
      "Climate Data Science and Remote Sensing",
      "Analytics Application Development",
      "Neural Operators (FNO, DeepONet)",
    ],
  },
];

const getIconContainerClass = (color: string) => {
  switch (color) {
    case "cool":
      return "icon-container-cool";
    case "warm":
      return "icon-container-warm";
    case "primary":
      return "icon-container-primary";
    case "violet":
      return "icon-container-primary";
    default:
      return "icon-container-primary";
  }
};

export default function CVPage() {
  return (
    <div className="py-12 md:py-20">
      <div className="container-default">
        {/* Header */}
        <div className="max-w-2xl mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
            Curriculum Vitae
          </h1>
          <p className="text-lg text-muted-foreground mb-6">
            Climate scientist and ML researcher with expertise in physics-informed
            machine learning, neural operators, and satellite remote sensing.
          </p>

          {/* Download Button */}
          <div className="flex flex-wrap gap-4">
            <a
              href="/cv.pdf"
              download
              className="btn btn-primary"
            >
              <Download className="h-5 w-5" />
              Download CV (PDF)
            </a>
            <Link href="/about" className="btn btn-secondary">
              <ExternalLink className="h-5 w-5" />
              Full About Page
            </Link>
          </div>
        </div>

        {/* Highlights Grid */}
        <div className="grid gap-6 md:grid-cols-2 mb-12">
          {highlights.map((section) => (
            <div key={section.title} className="card">
              <div className="flex items-center gap-3 mb-4">
                <div className={getIconContainerClass(section.color)}>
                  <section.icon className="h-5 w-5" />
                </div>
                <h2 className="text-xl font-semibold text-foreground">
                  {section.title}
                </h2>
              </div>
              <ul className="space-y-2">
                {section.items.map((item, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-muted-foreground"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-4 mb-12">
          {[
            { label: "Years Experience", value: "10+", color: "primary" },
            { label: "Publications", value: "5", color: "teal" },
            { label: "Books Authored", value: "2", color: "orange" },
          ].map((stat) => (
            <div
              key={stat.label}
              className="text-center p-6 rounded-[16px] bg-card shadow-soft"
            >
              <p className={`text-3xl font-bold mb-1 text-${stat.color}`}>{stat.value}</p>
              <p className="text-sm text-muted-foreground">{stat.label}</p>
            </div>
          ))}
        </div>

        {/* Key Skills */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-foreground mb-6">
            Key Technical Skills
          </h2>
          <div className="flex flex-wrap gap-2">
            {[
              { skill: "Python", type: "primary" },
              { skill: "PyTorch", type: "primary" },
              { skill: "TensorFlow", type: "primary" },
              { skill: "R", type: "primary" },
              { skill: "Neural Operators", type: "warm" },
              { skill: "FNO", type: "warm" },
              { skill: "PINNs", type: "warm" },
              { skill: "Google Earth Engine", type: "cool" },
              { skill: "QGIS", type: "cool" },
              { skill: "ERA5", type: "cool" },
              { skill: "CMIP6", type: "cool" },
              { skill: "ECOSTRESS", type: "cool" },
              { skill: "Remote Sensing", type: "violet" },
              { skill: "Statistical Downscaling", type: "violet" },
              { skill: "GIS", type: "violet" },
              { skill: "PostGIS", type: "violet" },
              { skill: "Docker", type: "muted" },
              { skill: "AWS", type: "muted" },
              { skill: "GCP", type: "muted" },
            ].map((item) => (
              <span
                key={item.skill}
                className={`badge badge-sm badge-${item.type}`}
              >
                {item.skill}
              </span>
            ))}
          </div>
        </div>

        {/* CTA */}
        <div className="text-center py-12 px-6 rounded-[16px] bg-card shadow-soft">
          <h2 className="text-2xl font-bold text-foreground mb-4">
            Want to know more?
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto mb-6">
            Check out my research publications, explore my projects, or get in
            touch for collaborations.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link href="/research" className="btn btn-primary">
              View Publications
            </Link>
            <Link href="/contact" className="btn btn-secondary">
              Contact Me
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
