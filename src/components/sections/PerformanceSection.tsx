import React from "react";
import Header from "../common/Header";
import "./PerformanceSection.css";

function PerformanceSection() {
  const performanceData = [
    {
      id: 0,
      metricName: "Accuracy",
      value: "98.7%",
      description:
        "The model correctly identifies fraudulent and non-fraudulent transactions with high precision.",
      imageUrl: "/images/accuracy.png",
    },
    {
      id: 1,
      metricName: "Precision",
      value: "97.5%",
      description:
        "The percentage of transactions classified as fraudulent that were indeed fraudulent.",
      imageUrl: "/images/precision.png",
    },
    {
      id: 2,
      metricName: "Recall",
      value: "96.3%",
      description:
        "The model successfully detected the majority of fraudulent transactions.",
      imageUrl: "/images/recall.png",
    },
    {
      id: 3,
      metricName: "F1-Score",
      value: "96.9%",
      description:
        "The harmonic mean of precision and recall ensures a balanced measure of the model's performance.",
      imageUrl: "/images/f1score.png",
    },
    {
      id: 4,
      metricName: "Detection Time",
      value: "1.2 seconds",
      description:
        "Average time taken by the HGNN model to evaluate and classify a transaction.",
      imageUrl: "/images/detection-time.png",
    },
  ];

  return (
    <section className="performance-section">
      <Header
        title="Model Performance Metrics"
        subtitle="Quantifying the effectiveness of our payment fraud detection system"
      />
      <div className="metrics-container">
        {performanceData.map((metric) => (
          <div key={metric.id} className="metric-card">
            <img
              src={metric.imageUrl}
              alt={metric.metricName}
              className="metric-image"
            />
            <h3 className="metric-name">{metric.metricName}</h3>
            <p className="metric-value">{metric.value}</p>
            <p className="metric-description">{metric.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

export default PerformanceSection;
