import React from "react";
import Header from "../common/Header";
import "./PerformanceSection.css";
import Link from "next/link";

function PerformanceSection() {
  const performanceData = [
    {
      id: 0,
      value: "96.3%",
      metricName: "Accuracy",
      description:
        "The model correctly identifies fraudulent and non-fraudulent transactions with high precision.",
      imageUrl: "/images/accuracy.png",
    },

    {
      id: 1,
      value: "96.3%",
      metricName: "Precision",
      description:
        "The percentage of transactions classified as fraudulent that were indeed fraudulent.",
      imageUrl: "/images/precision.png",
    },

    {
      id: 2,
      value: "96.3%",
      metricName: "Recall",
      description:
        "The model successfully detected the majority of fraudulent transactions.",
      imageUrl: "/images/recall.png",
    },

    {
      id: 3,
      value: "96.3%",
      metricName: "F1-Score",
      description:
        "The harmonic mean of precision and recall ensures a balanced measure of the model's performance.",
      imageUrl: "/images/f1score.png",
    },

    {
      id: 4,
      value: "96.3%",
      metricName: "Detection Time",
      description:
        "Average time taken by the GNN model to evaluate and classify a transaction.",
      imageUrl: "/images/detection-time.png",
    },

    {
      id: 5,
      value: "96.3%",
      metricName: "Detection Time",
      description:
        "Average time taken by the GNN model to evaluate and classify a transaction.",
      imageUrl: "/images/detection-time.png",
    },

  ];

  return (
    <section className="performance-section">

      <Header
        title="Model Performance Metrics"
        subtitle="Quantifying the effectiveness of our payment fraud detection system"
      /> 

      <Link href="/model-metrics" className="model-m-link">
          Model Metrics →
      </Link>

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