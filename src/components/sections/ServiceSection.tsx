import React from "react";
import Header from "../common/Header";
import ServiceCard from "../cards/ServiceCard";
import "./ServiceSection.css";

function ServiceSection() {
  const serviceData = [

    {
      id: 0,
      iconUrl: "/images/risk_analysis_icon.png",
      title: "Risk Scoring & Analysis",
      description:
        "Automatically assess transaction risks with advanced AI-driven risk scoring to proactively mitigate potential fraud threats.",
    },

    {
      id: 1,
      iconUrl: "/images/behavior_analysis_icon.png",
      title: "Behavioral Analysis",
      description:
        "Analyze user behaviors and payment patterns to differentiate between legitimate users and potential fraudsters.",
    },
    {
      id: 2,
      iconUrl: "/images/data_visualization_icon.png",
      title: "Data Visualization & Insights",
      description:
        "Get comprehensive dashboards and visual insights into fraud trends, transaction patterns, and security metrics.",
    },
  ];

  return (
    <section className="service-section">
      <Header title="service" subtitle="Our Vision & Our Goal" />
      <div className="service-grid">
        {serviceData.map((service) => (
          <ServiceCard
            key={service.id}
            iconUrl={service.iconUrl}
            title={service.title}
            description={service.description}
            cardClass="custom-card-class"
            iconClass="custom-icon-class" 
            titleClass="custom-title-class" 
            descriptionClass="custom-description-class"
          />
        ))}
      </div>
    </section>
  );
}

export default ServiceSection;