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
    degree: "MSc in Climate Science",
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
    degree: "Master of Development Studies",
    school: "University of Dhaka",
    years: "2011 - 2012",
    details: [],
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
    title: "Graduate Research Assistant",
    company: "George Mason University",
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
              <div className="relative aspect-square rounded-[16px] overflow-hidden border-2 border-border bg-muted shadow-soft">
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

            <div className="prose prose-lg max-w-none text-muted-foreground space-y-4">
              <p className="!mb-0">
                I am a graduate student completing MSc in Climate Science at{" "}
                <Link
                  href="https://www.gmu.edu/"
                  target="_blank"
                  className="text-primary hover:underline"
                >
                  George Mason University
                </Link>
                , with over a decade of experience in statistics, spatial data
                science, and climate consulting and research. My career has spanned roles as a
                statistician, spatial data scientist, and climate researcher - now
                converging on a single goal: building AI-driven solutions for
                climate and environmental challenges.
              </p>
              <p className="!mb-0">
                I hold an MA in Climate and Society from Columbia University and
                have authored two books on GIS and remote sensing. My peer-reviewed
                research spans climate extremes, aquaculture economics, and climate
                data tools. I&apos;ve also taught graduate-level courses and consulted
                on statistical and geospatial projects across sectors.
              </p>
              <p className="!mb-0">
                Currently, I&apos;m exploring how machine learning can advance climate
                prediction - from satellite-based urban heat mapping to building
                interactive tutorials that make AI research accessible. I aspire to
                develop practical AI tools that help communities understand and
                adapt to a changing climate.
              </p>
            </div>
          </div>
        </div>

        {/* Research Interests - Purple theme */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-8">
            <div className="icon-container-primary">
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
                className="card"
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

        {/* Education - Cool/Teal theme */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-8">
            <div className="icon-container-cool">
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
                className="timeline-item"
              >
                <div className="timeline-dot" style={{ backgroundColor: 'var(--teal)' }} />
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <h3 className="font-semibold text-foreground">{edu.degree}</h3>
                  <span className="badge badge-sm badge-cool">
                    <Calendar className="h-3 w-3" />
                    {edu.years}
                  </span>
                </div>
                <p className="text-teal font-medium mb-2">{edu.school}</p>
                <ul className="text-sm text-muted-foreground space-y-1">
                  {edu.details.map((detail, i) => (
                    <li key={i}>- {detail}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* Experience - Warm/Orange theme */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-8">
            <div className="icon-container-warm">
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
                className="timeline-item"
              >
                <div className="timeline-dot timeline-dot-accent" />
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <h3 className="font-semibold text-foreground">{exp.title}</h3>
                  <span className="badge badge-sm badge-warm">
                    <Calendar className="h-3 w-3" />
                    {exp.years}
                  </span>
                </div>
                <p className="text-orange font-medium mb-2">{exp.company}</p>
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
            <div className="icon-container-primary">
              <Code className="h-6 w-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Technical Skills
            </h2>
          </div>

          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            <div className="card-static">
              <h3 className="font-semibold text-foreground mb-3">Languages</h3>
              <div className="flex flex-wrap gap-2">
                {skills.languages.map((skill) => (
                  <span
                    key={skill}
                    className="badge badge-sm badge-primary"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>

            <div className="card-static">
              <h3 className="font-semibold text-foreground mb-3">ML/DL</h3>
              <div className="flex flex-wrap gap-2">
                {skills.ml.map((skill) => (
                  <span
                    key={skill}
                    className="badge badge-sm badge-warm"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>

            <div className="card-static">
              <h3 className="font-semibold text-foreground mb-3">Geospatial</h3>
              <div className="flex flex-wrap gap-2">
                {skills.geospatial.map((skill) => (
                  <span
                    key={skill}
                    className="badge badge-sm badge-violet"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>

            <div className="card-static">
              <h3 className="font-semibold text-foreground mb-3">Climate Data</h3>
              <div className="flex flex-wrap gap-2">
                {skills.climate.map((skill) => (
                  <span
                    key={skill}
                    className="badge badge-sm badge-cool"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="text-center py-12 px-6 rounded-[16px] bg-card shadow-soft">
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
