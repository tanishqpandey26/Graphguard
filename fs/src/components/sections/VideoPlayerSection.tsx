"use client"; 

import React, { useState, useEffect } from "react";
import "./VideoPlayerSection.css";

function VideoPlayerSection() {
  
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  return (
    <section className="video-player-section">
      <div className="content-wrapper">
        <p className="title">GNN-based Payment Fraud Detection</p>
        <p className="description">
          Explore how our Graph Neural Network (GNN) model detects payment
          fraud with high accuracy, leveraging graph-based patterns and advanced
          AI techniques.
        </p>
      </div>
      <div className="video-wrapper">
        {isClient && (
          <video src="/images/gnn.mp4" controls className="video-player">
            Your browser does not support the video tag.
          </video>
        )}
      </div>
    </section>
  );
}

export default VideoPlayerSection;
