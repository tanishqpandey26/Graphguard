import React from "react";
import { Linkedin, Github, Twitter, Mail } from "lucide-react";
import "./TeamStyles.css";

const teamMembers = [
  {
    name: "Tanishq Pandey",
    role: "Full Stack Developer",
    description: "Passionate about building user-friendly web applications with modern technologies.",
    linkedin: "https://www.linkedin.com/in/pandey26tanishq/",
    github: "https://github.com/tanishqpandey26",
    twitter: "https://twitter.com/johndoe",
    email: "mailto:johndoe@example.com",
    image: "/images/tanishq.jpeg",
  },
  {
    name: "Devesh Singh Negi",
    role: "AI/ML Engineer",
    description: "Specializes in scalable backend solutions and database management.",
    linkedin: "https://linkedin.com/in/janesmith",
    github: "https://github.com/janesmith",
    twitter: "https://twitter.com/janesmith",
    email: "mailto:johndoe@example.com",
    image: "/images/devesh.jpeg",
  },
  {
    name: "Devom Bhatt",
    role: "Data Engineer",
    description: "Loves crafting beautiful and intuitive user experiences. bhatt juuuuuuuuuu",
    linkedin: "https://linkedin.com/in/alicebrown",
    github: "https://github.com/alicebrown",
    twitter: "https://twitter.com/alicebrown",
    email: "mailto:johndoe@example.com",
    image: "/images/devom.jpeg",
  },
  {
    name: "Sanjana Verma",
    role: "System Engineer",
    description: "Loves crafting beautiful and intuitive user experiences. bhatt juuuuuuuuuu",
    linkedin: "https://linkedin.com/in/alicebrown",
    github: "https://github.com/alicebrown",
    twitter: "https://twitter.com/alicebrown",
    email: "mailto:johndoe@example.com",
    image: "/images/sanjana.jpeg",
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
