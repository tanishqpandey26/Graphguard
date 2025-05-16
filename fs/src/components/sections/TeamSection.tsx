import React from "react";
import { Linkedin, Github, Twitter, Mail } from "lucide-react";
import "./TeamStyles.css";

const teamMembers = [

  {
    name: "Devom Bhatt",
    role: "Data Analyst",
    description: "Transforms raw data into actionable insights. Passionate about uncovering trends through visual storytelling.",
    linkedin: "https://linkedin.com/in/alicebrown",
    github: "https://github.com/alicebrown",
    twitter: "https://twitter.com/alicebrown",
    email: "mailto:johndoe@example.com",
    image: "/images/devom.jpeg",
  },

  {
    name: "Sanjana Verma",
    role: "QA Engineer",
    description: "Ensures software quality with a sharp eye for detail. Loves breaking things to make them better.",
    linkedin: "https://linkedin.com/in/alicebrown",
    github: "https://github.com/alicebrown",
    twitter: "https://twitter.com/alicebrown",
    email: "mailto:johndoe@example.com",
    image: "/images/sanjana.jpeg",
  },

  {
    name: "Tanishq Pandey",
    role: "Full Stack Developer",
    description: "Builds seamless web experiences from front to back. Thrives on turning ideas into scalable solutions.",
    linkedin: "https://www.linkedin.com/in/pandey26tanishq/",
    github: "https://github.com/tanishqpandey26",
    twitter: "https://x.com/tanishqvatsa26",
    email: "mailto:tanishqpandeyofficial@gmail.com",
    image: "/images/tanishq.jpeg",
  },

  {
    name: "Devesh Singh Negi",
    role: "AI/ML Engineer",
    description: "Designs intelligent systems powered by machine learning. Loves solving problems with data and models.",
    linkedin: "https://linkedin.com/in/janesmith",
    github: "https://github.com/janesmith",
    twitter: "https://twitter.com/janesmith",
    email: "mailto:johndoe@example.com",
    image: "/images/devesh.jpeg",
  },
  
];

function TeamSection() {
  return (

    <section className="team-section">
      <h2>Meet Our Team</h2>

      <div className="team-container">

        {teamMembers.map((member, index) => (

          <div key={index} className="team-card">

            <img src={member.image} alt={member.name} className="team-image" />
            <h3>{member.name}</h3>

            <p className="team-role">{member.role}</p>
            <p className="team-description">{member.description}</p>

            <div className="team-socials">

              <a href={member.linkedin} target="_blank" rel="noopener noreferrer" title="linkedin">
                <Linkedin />
              </a>

              <a href={member.github} target="_blank" rel="noopener noreferrer" title="github">
                <Github />
              </a>

              <a href={member.twitter} target="_blank" rel="noopener noreferrer" title="twitter">
                <Twitter />
              </a>

              <a href={member.email} target="_blank" rel="noopener noreferrer" title="mail">
                <Mail />
              </a>

            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default TeamSection;
