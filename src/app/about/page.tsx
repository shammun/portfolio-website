import Image from "next/image";
import Link from "next/link";
import type { Metadata } from "next";
import {
  GraduationCap,
  Briefcase,
  Code,
  Brain,
  MapPin,
  Calendar,
  ExternalLink,
} from "lucide-react";

export const metadata: Metadata = {
  title: "About",
  description:
    "Climate scientist and ML researcher focusing on physics-informed machine learning, neural operators, and urban heat island prediction.",
};

const researchInterests = [
  {
    title: "Physics-Informed ML",
    description: "Developing neural networks that incorporate physical constraints for climate applications",
  },
  {
    title: "Urban Heat Islands",
    description: "Using satellite data (ECOSTRESS, Landsat) to map and predict urban thermal patterns",
  },
  {
    title: "Neural Operators",
    description: "Applying FNO and related architectures to climate modeling problems",
  },
  {
    title: "Climate Downscaling",
    description: "Bridging the gap between global climate models (CMIP6) and local impacts",
  },
  {
    title: "Foundation Models",
    description: "Exploring large-scale AI models for weather and climate prediction",
  },
  {
    title: "Extreme Events",
    description: "Using Graph Neural Networks and remote sensing for extreme event forecasting",
  },
];

const education = [
  {
    degree: "MS in Climate Science",
    school: "George Mason University",
    years: "2023 - Present",
    details: [
      "Concentration: Climate Data and Climate Modeling",
      "Thesis: Predicting Land Surface Temperature using Machine Learning and Satellite Remote Sensing",
    ],
  },
  {
    degree: "MA in Climate and Society",
    school: "Columbia University",
    years: "2013 - 2014",
    details: [
      "Full scholarship recipient",
      "Research with Dr. Andrew Robertson at IRI on decadal prediction",
    ],
  },
  {
    degree: "BSc in Statistics",
    school: "Shahjalal University of Science & Technology",
    years: "2002 - 2008",
    details: ["Government scholarship recipient"],
  },
];

const experience = [
  {
    title: "Local Climate Action Fellow",
    company: "Virginia Climate Center",
    years: "2024 - Present",
    description: "Statistically downscaling CMIP6 projections for Virginia, USA",
  },
  {
    title: "Research Assistant / Atmospheric Modeler",
    company: "George Mason University",
    years: "2023 - 2024",
    description: "ClimateIQ project (Google-funded): Urban climate modeling for NYC, Phoenix, and global cities",
  },
  {
    title: "Adjunct Faculty",
    company: "Jahangirnagar University",
    years: "2021 - 2023",
    description: "Taught Machine Learning and Python Programming for MS in Remote Sensing",
  },
  {
    title: "Country Consultant",
    company: "IRI, Columbia University",
    years: "2019 - 2022",
    description: "ACToday Project: NLP for climate damage extraction, seasonal forecast verification",
  },
];

const skills = {
  languages: ["Python", "R", "JavaScript", "C/C++", "SQL"],
  ml: ["PyTorch", "TensorFlow", "FNO", "PINNs", "GNNs", "LSTMs"],
  geospatial: ["Google Earth Engine", "QGIS", "ArcGIS", "PostGIS"],
  climate: ["ERA5", "CMIP6", "ECOSTRESS", "Landsat", "MODIS"],
};

export default function AboutPage() {
  return (
    <div className="py-12 md:py-20">
      <div className="container-default">
        {/* Header */}
        <div className="grid gap-8 lg:grid-cols-3 lg:gap-12 mb-16">
          {/* Photo */}
          <div className="lg:col-span-1">
            <div className="relative aspect-square max-w-sm mx-auto lg:mx-0">
              <div className="absolute inset-0 bg-primary/20 rounded-2xl blur-2xl" />
              <div className="relative aspect-square rounded-2xl overflow-hidden border-2 border-border bg-muted">
                <Image
                  src="/images/headshot.jpg"
                  alt="Shammunul Islam"
                  fill
                  className="object-cover"
                  priority
                />
              </div>
            </div>
          </div>

          {/* Bio */}
          <div className="lg:col-span-2">
            <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
              About Me
            </h1>
            <div className="flex items-center gap-2 text-muted-foreground mb-6">
              <MapPin className="h-4 w-4" />
              <span>Virginia, USA</span>
            </div>

            <div className="prose prose-lg max-w-none text-muted-foreground">
              <p>
                I am a climate scientist and machine learning researcher currently
                completing my MS in Climate Science at{" "}
                <Link
                  href="https://www.gmu.edu/"
                  target="_blank"
                  className="text-primary hover:underline"
                >
                  George Mason University
                </Link>
                . My research focuses on using physics-informed machine learning
                to predict urban land surface temperatures, combining satellite
                remote sensing with neural operators.
              </p>
              <p>
                With over a decade of experience spanning climate research,
                geospatial data science, and statistical consulting, I bring a
                unique interdisciplinary perspective to solving complex
                environmental challenges. I am the author of two international
                books on GIS and remote sensing, and have contributed to
                peer-reviewed research on climate extremes, aquaculture economics,
                and climate data tools.
              </p>
            </div>
          </div>
        </div>

        {/* Research Interests */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              <Brain className="h-6 w-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Research Interests
            </h2>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {researchInterests.map((interest) => (
              <div
                key={interest.title}
                className="p-4 rounded-xl border border-border bg-card hover:border-primary/50 transition-colors"
              >
                <h3 className="font-semibold text-foreground mb-2">
                  {interest.title}
                </h3>
                <p className="text-sm text-muted-foreground">
                  {interest.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Education */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              <GraduationCap className="h-6 w-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Education
            </h2>
          </div>

          <div className="space-y-6">
            {education.map((edu) => (
              <div
                key={edu.degree}
                className="relative pl-6 border-l-2 border-border hover:border-primary transition-colors"
              >
                <div className="absolute -left-2 top-0 w-4 h-4 rounded-full bg-primary" />
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <h3 className="font-semibold text-foreground">{edu.degree}</h3>
                  <span className="text-sm text-muted-foreground">
                    <Calendar className="inline h-3 w-3 mr-1" />
                    {edu.years}
                  </span>
                </div>
                <p className="text-primary font-medium mb-2">{edu.school}</p>
                <ul className="text-sm text-muted-foreground space-y-1">
                  {edu.details.map((detail, i) => (
                    <li key={i}>• {detail}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* Experience */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              <Briefcase className="h-6 w-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Experience
            </h2>
          </div>

          <div className="space-y-6">
            {experience.map((exp) => (
              <div
                key={exp.title}
                className="relative pl-6 border-l-2 border-border hover:border-primary transition-colors"
              >
                <div className="absolute -left-2 top-0 w-4 h-4 rounded-full bg-accent" />
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <h3 className="font-semibold text-foreground">{exp.title}</h3>
                  <span className="text-sm text-muted-foreground">
                    <Calendar className="inline h-3 w-3 mr-1" />
                    {exp.years}
                  </span>
                </div>
                <p className="text-primary font-medium mb-2">{exp.company}</p>
                <p className="text-sm text-muted-foreground">{exp.description}</p>
              </div>
            ))}
          </div>

          <div className="mt-6">
            <Link
              href="/cv"
              className="inline-flex items-center gap-2 text-primary font-medium hover:underline"
            >
              View Full CV
              <ExternalLink className="h-4 w-4" />
            </Link>
          </div>
        </section>

        {/* Skills */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              <Code className="h-6 w-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Technical Skills
            </h2>
          </div>

          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <h3 className="font-semibold text-foreground mb-3">Languages</h3>
              <div className="flex flex-wrap gap-2">
                {skills.languages.map((skill) => (
                  <span
                    key={skill}
                    className="px-3 py-1 text-sm rounded-full bg-primary/10 text-primary"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-foreground mb-3">ML/DL</h3>
              <div className="flex flex-wrap gap-2">
                {skills.ml.map((skill) => (
                  <span
                    key={skill}
                    className="px-3 py-1 text-sm rounded-full bg-accent/10 text-accent"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-foreground mb-3">Geospatial</h3>
              <div className="flex flex-wrap gap-2">
                {skills.geospatial.map((skill) => (
                  <span
                    key={skill}
                    className="px-3 py-1 text-sm rounded-full bg-blue-500/10 text-blue-500"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-foreground mb-3">Climate Data</h3>
              <div className="flex flex-wrap gap-2">
                {skills.climate.map((skill) => (
                  <span
                    key={skill}
                    className="px-3 py-1 text-sm rounded-full bg-green-500/10 text-green-500"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="text-center py-12 px-6 rounded-2xl bg-muted">
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-4">
            Interested in Collaboration?
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto mb-6">
            I&apos;m open to research collaborations, speaking opportunities, and
            consulting projects related to AI for climate applications.
          </p>
          <Link href="/contact" className="btn btn-primary">
            Get in Touch
          </Link>
        </section>
      </div>
    </div>
  );
}
