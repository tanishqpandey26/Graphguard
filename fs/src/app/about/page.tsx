import React from "react";
import Navbar from "@/components/common/Navbar";
import FooterSection from "@/components/sections/FooterSection";
import Link from 'next/link';
import './AboutStyles.css';

function About() {
  return (
    <>   
        <Navbar />

         <section className="aboutSection">
  <div className="container">
    <h1 className="heading">About GraphGuard</h1>
    <p className="description">
      GraphGuard is an intelligent system built using Graph Neural Networks (GNNs) to detect credit card fraud in real-time. 
      It offers robust analytics and model insights through an intuitive interface, enabling businesses to stay ahead of fraudulent activities 
      with cutting-edge graph-based machine learning.
    </p>

    <div className="routesInfo">
      <h2 className="subheading">Explore the Site</h2>
      <ul className="routesList">

        <li>
          <Link href="/docs" className="link">
            Documentation
          </Link>
          <p>
            Understand the complete data pipeline, model architecture, and implementation steps. 
            This section offers developer-friendly guidance on integrating GraphGuard into existing systems, 
            along with detailed API references and setup instructions.
          </p>
        </li>

        <li>
          <Link href="/model-metrics" className="link">
            Model Metrics
          </Link>
          <p>
            Dive into rich visualizations and interactive graphs that showcase the models performance. 
            Analyze metrics such as precision, recall, F-score, and real-time anomaly detection capabilities 
            through intuitive dashboards.
          </p>
        </li>

        <li>
          <Link href="/dashboard" className="link">
            Test Dashboard
          </Link>
          <p>
            Upload your own transaction dataset in CSV format to evaluate GraphGuards detection accuracy. 
            Experience live fraud detection using our trained GNN model and see flagged anomalies in real-time.
          </p>
        </li>

        <li>
          <Link href="/team" className="link">
            Meet the Team
          </Link>
          <p>
            Get to know the passionate engineers and researchers behind GraphGuard. 
            Learn about our backgrounds and areas of expertise.
          </p>
        </li>

      </ul>
    </div>
  </div>
</section>


        <FooterSection />
    </> 
  );
}

export default About;
