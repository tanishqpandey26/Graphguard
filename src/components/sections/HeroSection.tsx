import React from "react";
import MainButton from "../common/MainButton";
import "./HeroSection.css";

function HeroSection() {
  return (
    <section className="hero-section">
    
      <div className="hero-content">
        <p className="hero-title">
          Credit Card Fraud Detection using HGNN
        </p>
        <p className="hero-subtitle">
          Fraud Detection Powered by Heterogeneous Graph Neural Networks (HGNNs).
        </p>
        <div className="hero-buttons">
          <button className="hero-main-button">Get Started</button>
          <div className="hero-learn-more">
            <span>Learn More</span>
          </div>
        </div>
      </div>

      
      <div className="hero-image">
        <img
          src="/images/secure-pay.png"
          alt="Guy with phone surrounded by action icons"
        />
      </div>
    </section>
  );
}

export default HeroSection;