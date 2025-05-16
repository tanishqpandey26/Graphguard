import React from "react";

function ProjectDocs() {
  return (
    <>

      <section className="px-6 md:px-16 py-12 bg-gray-50 min-h-screen">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-6">
            GraphGuard: Model Metrics & Visualization
          </h1>

          <p className="text-gray-600 text-lg leading-relaxed mb-8">
            Welcome to the <span className="font-semibold text-blue-600">Model Metrics</span> section of GraphGuard – our credit card fraud detection system powered by Graph Neural Networks (GNNs).
            The carousel above contains navigational links to explore various facets of our project:
          </p>

          <ul className="list-disc list-inside text-gray-700 text-base space-y-2 mb-8">
            <li><strong>Dataset Insights</strong> – Data distributions, class imbalance, preprocessing.</li>
            <li><strong>Model Parameters</strong> – Training configs, epochs, learning rate, etc.</li>
            <li><strong>Node & Tree Visualizations</strong> – Graph structures and relationships.</li>
            <li><strong>Statistical Graphs</strong> – Accuracy, loss, ROC curves, and evaluation metrics.</li>
          </ul>

          <p className="text-gray-600 text-lg mb-8">
            Navigate through the carousel to explore detailed visual analytics and technical metrics of our AI-powered fraud detection engine.
          </p>

          <div className="space-y-4">
            <a
              href="/"
              className="inline-block bg-blue-500 text-white px-6 py-3 rounded-md shadow hover:bg-blue-700 transition"
            >
               Download Processed CSV
            </a>

            <br />

            <a
              href="https://www.kaggle.com/datasets/kartik2112/fraud-detection"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block text-blue-600 font-medium underline hover:text-blue-800 transition"
            >
              🔗 View Original Dataset on Kaggle
            </a>
          </div>
        </div>
      </section>
    </>
  );
}

export default ProjectDocs;
