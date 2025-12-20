import Link from "next/link";
import type { Metadata } from "next";
import {
  BookOpen,
  FileText,
  ExternalLink,
  Calendar,
  Users,
  Award,
} from "lucide-react";

export const metadata: Metadata = {
  title: "Research & Publications",
  description:
    "Academic publications, books, and research by Shammunul Islam on climate science, machine learning, and remote sensing.",
};

const publications = [
  {
    type: "journal",
    title:
      "Climate change quadruples flood-causing extreme monsoon rainfall events in Bangladesh and northeast India",
    authors: [
      "Fahad, A. A.",
      "Hasan, M.",
      "Sharmili, N.",
      "Islam, S.",
      "Swenson, E. T.",
      "Roxy, M. K.",
    ],
    journal: "Quarterly Journal of the Royal Meteorological Society",
    year: 2024,
    volume: "150(760)",
    pages: "1267-1287",
    doi: "https://doi.org/10.1002/qj.4645",
    tags: ["Climate Change", "Extreme Rainfall", "Monsoon", "Flooding"],
  },
  {
    type: "journal",
    title:
      "Economic valuation of climate induced losses to aquaculture for evaluating climate information services in Bangladesh",
    authors: [
      "Islam, S.",
      "Hossain, P. R.",
      "Braun, M.",
      "Amjath-Babu, T. S.",
      "Mohammed, E. Y.",
      "Krupnik, T. J.",
      "Chowdhury, A. H.",
      "Thomas, M.",
      "Mauerman, M.",
    ],
    journal: "Climate Risk Management",
    year: 2024,
    volume: "43",
    pages: "Article 100582",
    doi: "https://doi.org/10.1016/j.crm.2023.100582",
    tags: ["Climate Services", "Aquaculture", "Economic Valuation"],
  },
  {
    type: "journal",
    title: "The climate data tool: Enhancing climate services across Africa",
    authors: [
      "Dinku, T.",
      "Faniriantsoa, R.",
      "Islam, S.",
      "Nsengiyumva, G.",
      "Grossi, A.",
    ],
    journal: "Frontiers in Climate",
    year: 2022,
    volume: "3",
    pages: "Article 787519",
    doi: "https://doi.org/10.3389/fclim.2021.787519",
    tags: ["Climate Services", "Africa", "Climate Data"],
  },
];

const books = [
  {
    title: "Hands-on Geospatial Analysis with R and QGIS",
    subtitle:
      "A beginner's guide to manipulating, managing, and analyzing spatial data using R and QGIS 3.2.2",
    authors: ["Shammunul Islam"],
    publisher: "Packt Publishing",
    year: 2018,
    isbn: "9781788991674",
    link: "https://www.packtpub.com/en-us/product/hands-on-geospatial-analysis-with-r-and-qgis-9781788991674",
    tags: ["GIS", "QGIS", "R", "Spatial Analysis"],
  },
  {
    title: "Mastering Geospatial Development with QGIS 3.x",
    subtitle:
      "An in-depth guide to becoming proficient in spatial data analysis using QGIS 3.X (3rd Edition)",
    authors: [
      "Shammunul Islam",
      "Simon Miles",
      "Kurt Menke",
      "Richard Smith Jr.",
      "Luigi Pirelli",
      "John Van Hoesen",
    ],
    publisher: "Packt Publishing",
    year: 2019,
    isbn: "9781788999892",
    link: "https://www.amazon.com/Mastering-Geospatial-Development-QGIS-depth/dp/1788999894",
    tags: ["GIS", "QGIS", "Geospatial Development"],
  },
];

const thesis = {
  title:
    "Predicting Land Surface Temperature using Machine Learning and Satellite Remote Sensing",
  type: "MS Thesis",
  university: "George Mason University",
  department: "Department of Atmospheric, Oceanic and Earth Sciences",
  year: 2025,
  status: "In Progress (Defense: November 2025)",
  abstract:
    "This thesis develops machine learning model to predict urban land surface temperature across New York City at 70-meter resolution. Using ECOSTRESS satellite observations, ERA5 reanalysis data, NDVI vegetation data, and urban morphology parameters, the model achieves superior spatial pattern accuracy compared to traditional modeling approaches.",
  tags: ["FNO", "Machine Learning", "Urban Heat Island", "ECOSTRESS"],
};

export default function ResearchPage() {
  return (
    <div className="py-12 md:py-20">
      <div className="container-default">
        {/* Header */}
        <div className="max-w-2xl mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
            Research &amp; Publications
          </h1>
          <p className="text-lg text-muted-foreground">
            My research focus on the application of AI in urban climate. My previous
            researches covered topics such as climate data analysis tool, measuring
            economic impact of extreme weather events using newspaper scraping and
            flood mapping.
          </p>
        </div>

        {/* Current Research - Primary/Purple theme */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <div className="icon-container-primary">
              <Award className="h-6 w-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Current Research
            </h2>
          </div>

          <div className="card-featured">
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <span className="badge badge-sm badge-primary">
                {thesis.type}
              </span>
              <span className="badge badge-sm badge-muted">
                <Calendar className="h-3 w-3" />
                {thesis.status}
              </span>
            </div>
            <h3 className="text-xl font-semibold text-foreground mb-2">
              {thesis.title}
            </h3>
            <p className="text-sm text-primary mb-3">
              {thesis.university} - {thesis.department}
            </p>
            <p className="text-muted-foreground mb-4">{thesis.abstract}</p>
            <div className="flex flex-wrap gap-2">
              {thesis.tags.map((tag) => (
                <span
                  key={tag}
                  className="badge badge-sm badge-primary"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </section>

        {/* Peer-Reviewed Publications - Warm/Orange theme */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <div className="icon-container-warm">
              <FileText className="h-6 w-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Peer-Reviewed Publications
            </h2>
          </div>

          <div className="space-y-6">
            {publications.map((pub, index) => (
              <div key={index} className="card">
                <div className="flex flex-wrap items-center gap-2 mb-2">
                  <span className="badge badge-sm badge-warm">
                    Journal Article
                  </span>
                  <span className="badge badge-sm badge-muted">
                    <Calendar className="h-3 w-3" />
                    {pub.year}
                  </span>
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-2">
                  {pub.title}
                </h3>
                <p className="text-sm text-muted-foreground mb-2">
                  <Users className="inline h-3 w-3 mr-1" />
                  {pub.authors.join(", ")}
                </p>
                <p className="text-sm text-orange font-medium mb-3">
                  <em>{pub.journal}</em>, {pub.volume}, {pub.pages}
                </p>
                <div className="flex flex-wrap items-center gap-4">
                  <a
                    href={pub.doi}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
                  >
                    <ExternalLink className="h-3 w-3" />
                    View Paper
                  </a>
                  <div className="flex flex-wrap gap-2">
                    {pub.tags.map((tag) => (
                      <span
                        key={tag}
                        className="badge badge-sm badge-muted"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Books - Cool/Teal theme */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <div className="icon-container-cool">
              <BookOpen className="h-6 w-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Books
            </h2>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            {books.map((book, index) => (
              <div key={index} className="card">
                <div className="flex flex-wrap items-center gap-2 mb-2">
                  <span className="badge badge-sm badge-cool">
                    Book
                  </span>
                  <span className="badge badge-sm badge-muted">
                    <Calendar className="h-3 w-3" />
                    {book.year}
                  </span>
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-1">
                  {book.title}
                </h3>
                <p className="text-sm text-muted-foreground mb-2">
                  {book.subtitle}
                </p>
                <p className="text-sm text-muted-foreground mb-2">
                  <Users className="inline h-3 w-3 mr-1" />
                  {book.authors.join(", ")}
                </p>
                <p className="text-sm text-teal font-medium mb-3">
                  {book.publisher} - ISBN: {book.isbn}
                </p>
                <div className="flex flex-wrap items-center gap-4">
                  <a
                    href={book.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
                  >
                    <ExternalLink className="h-3 w-3" />
                    View Book
                  </a>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* CTA */}
        <div className="text-center py-12 px-6 rounded-[16px] bg-card shadow-soft">
          <h2 className="text-2xl font-bold text-foreground mb-4">
            Interested in Collaboration?
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto mb-6">
            I&apos;m always looking for research collaborations in climate
            science and machine learning.
          </p>
          <Link href="/contact" className="btn btn-primary">
            Get in Touch
          </Link>
        </div>
      </div>
    </div>
  );
}
