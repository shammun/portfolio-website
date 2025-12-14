"use client";

import { useState } from "react";
import {
  Mail,
  MapPin,
  Github,
  Linkedin,
  GraduationCap,
  Send,
  CheckCircle,
  AlertCircle,
} from "lucide-react";

const socialLinks = [
  {
    name: "Email",
    value: "sha_is13@yahoo.com",
    href: "mailto:sha_is13@yahoo.com",
    icon: Mail,
  },
  {
    name: "Email (GMU)",
    value: "sislam27@gmu.edu",
    href: "mailto:sislam27@gmu.edu",
    icon: Mail,
  },
  {
    name: "LinkedIn",
    value: "linkedin.com/in/shammunul",
    href: "https://www.linkedin.com/in/shammunul/",
    icon: Linkedin,
  },
  {
    name: "GitHub",
    value: "github.com/shammun",
    href: "https://github.com/shammun",
    icon: Github,
  },
  {
    name: "Google Scholar",
    value: "View Profile",
    href: "https://scholar.google.com/citations?user=YOUR_ID",
    icon: GraduationCap,
  },
];

export default function ContactPage() {
  const [formState, setFormState] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: "",
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormState("loading");

    // Simulate form submission - replace with actual form handler (Formspree, etc.)
    try {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setFormState("success");
      setFormData({ name: "", email: "", subject: "", message: "" });
    } catch {
      setFormState("error");
    }
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  return (
    <div className="py-12 md:py-20">
      <div className="container-default">
        {/* Header */}
        <div className="max-w-2xl mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
            Get in Touch
          </h1>
          <p className="text-lg text-muted-foreground">
            I&apos;m always interested in research collaborations, speaking
            opportunities, and discussions about climate science and AI. Feel free
            to reach out!
          </p>
        </div>

        <div className="grid gap-12 lg:grid-cols-2">
          {/* Contact Form */}
          <div>
            <div className="card">
              <h2 className="text-xl font-semibold text-foreground mb-6">
                Send a Message
              </h2>

              {formState === "success" ? (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <div className="w-12 h-12 rounded-full bg-cool-light flex items-center justify-center mb-4">
                    <CheckCircle className="h-6 w-6 text-cool-text" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    Message Sent!
                  </h3>
                  <p className="text-muted-foreground mb-4">
                    Thank you for reaching out. I&apos;ll get back to you soon.
                  </p>
                  <button
                    onClick={() => setFormState("idle")}
                    className="btn btn-secondary"
                  >
                    Send Another Message
                  </button>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label
                      htmlFor="name"
                      className="block text-sm font-medium text-foreground mb-1"
                    >
                      Name *
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      required
                      value={formData.name}
                      onChange={handleChange}
                      className="w-full px-4 py-2 rounded-lg border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                      placeholder="Your name"
                    />
                  </div>

                  <div>
                    <label
                      htmlFor="email"
                      className="block text-sm font-medium text-foreground mb-1"
                    >
                      Email *
                    </label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      required
                      value={formData.email}
                      onChange={handleChange}
                      className="w-full px-4 py-2 rounded-lg border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                      placeholder="your.email@example.com"
                    />
                  </div>

                  <div>
                    <label
                      htmlFor="subject"
                      className="block text-sm font-medium text-foreground mb-1"
                    >
                      Subject *
                    </label>
                    <select
                      id="subject"
                      name="subject"
                      required
                      value={formData.subject}
                      onChange={handleChange}
                      className="w-full px-4 py-2 rounded-lg border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                    >
                      <option value="">Select a topic</option>
                      <option value="research">Research Collaboration</option>
                      <option value="speaking">Speaking Opportunity</option>
                      <option value="consulting">Consulting Project</option>
                      <option value="tutorial">Question about Tutorials</option>
                      <option value="other">Other</option>
                    </select>
                  </div>

                  <div>
                    <label
                      htmlFor="message"
                      className="block text-sm font-medium text-foreground mb-1"
                    >
                      Message *
                    </label>
                    <textarea
                      id="message"
                      name="message"
                      required
                      rows={5}
                      value={formData.message}
                      onChange={handleChange}
                      className="w-full px-4 py-2 rounded-lg border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary resize-none"
                      placeholder="Your message..."
                    />
                  </div>

                  {formState === "error" && (
                    <div className="flex items-center gap-2 text-red-500 text-sm">
                      <AlertCircle className="h-4 w-4" />
                      Something went wrong. Please try again.
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={formState === "loading"}
                    className="btn btn-primary w-full"
                  >
                    {formState === "loading" ? (
                      <>
                        <span className="animate-spin">...</span>
                        Sending...
                      </>
                    ) : (
                      <>
                        <Send className="h-4 w-4" />
                        Send Message
                      </>
                    )}
                  </button>
                </form>
              )}
            </div>
          </div>

          {/* Contact Info */}
          <div className="space-y-8">
            {/* Location */}
            <div className="card">
              <div className="flex items-center gap-3 mb-4">
                <div className="icon-container-primary">
                  <MapPin className="h-5 w-5" />
                </div>
                <h2 className="text-xl font-semibold text-foreground">
                  Location
                </h2>
              </div>
              <p className="text-muted-foreground">
                Virginia, USA
                <br />
                <span className="text-sm">
                  George Mason University, Department of Atmospheric, Oceanic and
                  Earth Sciences
                </span>
              </p>
            </div>

            {/* Contact Links */}
            <div className="card">
              <h2 className="text-xl font-semibold text-foreground mb-4">
                Connect With Me
              </h2>
              <div className="space-y-3">
                {socialLinks.map((link) => (
                  <a
                    key={link.name}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-3 p-3 rounded-lg hover:bg-muted transition-colors group hover:no-underline"
                  >
                    <div className="icon-container-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                      <link.icon className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="font-medium text-foreground">{link.name}</p>
                      <p className="text-sm text-muted-foreground">{link.value}</p>
                    </div>
                  </a>
                ))}
              </div>
            </div>

            {/* Open To */}
            <div className="card-featured">
              <h2 className="text-xl font-semibold text-foreground mb-4">
                Open To
              </h2>
              <ul className="space-y-2 text-muted-foreground">
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-teal" />
                  Research Collaborations
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-orange" />
                  Speaking Opportunities
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-primary" />
                  Consulting Projects
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-primary" />
                  PhD Opportunities
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
