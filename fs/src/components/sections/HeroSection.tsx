import React from "react";
import "./HeroSection.css";
import Link from 'next/link';

function HeroSection() {
  return (
    <section className="hero-section">
    
      <div className="hero-content">
        <p className="hero-title">
          Credit Card Fraud Detection using GNN
        </p>
        <p className="hero-subtitle">
        Detecting fraudulent activities in real-time requires cutting-edge technology. Our solution leverages Graph Neural Networks (GNNs) to uncover hidden patterns across complex transaction networks, ensuring unparalleled accuracy in identifying credit card fraud. 
        </p>
        <div className="hero-buttons">

          <Link href="/about">      
              <button className="hero-main-button" >Get Started
          </button>
          </Link>
          
          <Link href="/docs">
          <div className="hero-learn-more">
            <span>Learn More</span>
          </div>
          </Link>

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
