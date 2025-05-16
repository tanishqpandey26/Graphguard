import React from "react";
import "./DocsStyles.css";
import Link from "next/link";


function DocsSection() {
  
  return (

    <section className="docs-section">

      <div className="docs-content">
        <h2>Explore GraphGuard Documentation</h2>
        <p>
          Learn more about how GraphGuard detects payment fraud using advanced AI and graph-based techniques. Dive into the documentation for detailed insights.
        </p>

        <Link href="/docs" className="docs-link">
          Read Documentation →
        </Link>

      </div>

      <div className="docs-image">
        <img src="/images/docs.jpg" alt="GraphGuard Documentation Preview" />
      </div>

    </section>
  );
}

export default DocsSection;
