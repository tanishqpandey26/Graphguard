import React from "react";
import Link from "next/link";
import Navbar  from '@/components/common/Navbar';
import FooterSection from "@/components/sections/FooterSection";

function ProjectDocs() {
  return (
    <>   
        <Navbar />

        <section className="px-6 py-16 max-w-6xl mx-auto mt-auto mt-5">
      <h1 className="text-4xl md:text-5xl font-bold text-darkBlue mb-6">
        Documentation
      </h1>

      <p className="text-lg text-gray-600 mb-8">
        Welcome to the documentation for our{" "}
        <span className="text-lightBlue font-semibold">
          GNN-based Payment Fraud Detection
        </span>{" "}
        platform. This platform utilizes a hypergraph neural network (GNN) to
        analyze transaction networks and detect fraudulent activity with
        high precision.
      </p>

      <div className="grid gap-8 md:grid-cols-2">
        <div className="bg-white p-6 rounded-xl shadow-lg border hover:shadow-xl transition-shadow duration-300">
          <h2 className="text-2xl font-semibold text-darkBlue mb-4">
            How it Works
          </h2>
          <p className="text-gray-600">
            We represent users, transactions, and merchants as nodes in a
            hypergraph. By modeling complex relationships among them, our
            GNN learns meaningful embeddings and classifies transactions as
            fraudulent or genuine.
          </p>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-lg border hover:shadow-xl transition-shadow duration-300">
          <h2 className="text-2xl font-semibold text-darkBlue mb-4">
            Key Features
          </h2>
          <ul className="list-disc pl-5 text-gray-600 space-y-2">
            <li>Real-time fraud detection</li>
            <li>Hypergraph neural network architecture</li>
            <li>Graph-based insights & feature propagation</li>
            <li>High model accuracy and precision</li>
          </ul>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-lg border hover:shadow-xl transition-shadow duration-300">
          <h2 className="text-2xl font-semibold text-darkBlue mb-4">
            Getting Started
          </h2>
          <p className="text-gray-600 mb-4">
            Start by signing in and navigating to the dashboard. From there,
            upload transaction datasets or integrate via API to begin
            predictions. Make sure to check the model metrics regularly to
            evaluate system performance.
          </p>
          <Link
            href="/model-metrics"
            className="text-lightBlue font-medium hover:underline"
          >
            View Model Metrics →
          </Link>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-lg border hover:shadow-xl transition-shadow duration-300">
          <h2 className="text-2xl font-semibold text-darkBlue mb-4">
            API Usage
          </h2>
          <p className="text-gray-600">
            Use our REST API to submit transaction data. The API accepts JSON
            payloads and returns fraud probability scores along with
            classification results. API documentation and tokens can be found
            in your profile settings.
          </p>
        </div>
      </div>
    </section>

        <FooterSection />
    </>
  );
}

export default ProjectDocs;
