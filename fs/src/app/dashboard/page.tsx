'use client';

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import Navbar from "@/components/common/Navbar";
import "./DashboardStyles.css";
import { FilePlus } from 'lucide-react';

function Dashboard() {

  const [fileName, setFileName] = useState<string | null>(null);
  
  const router = useRouter();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      setTimeout(() => {
        router.push("/dashboard/loading");
      }, 3000);
    }
  };

  return (
    <>
      <Navbar />

      <section className="dashboardSection">
        <div className="container">
          <h1 className="heading">Upload Your CSV File</h1>
          <p className="description">
            Select a CSV file to simulate model testing. You will be routed to query form after processing is completed.
          </p>

         <label className="uploadBox">
  <input
    type="file"
    accept=".csv"
    onChange={handleFileChange}
    className="input"
  />
  {fileName ? (
    <span className="fileName">{fileName}</span>
  ) : (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem" }}>
      <FilePlus size={20} />
      Upload .csv file
    </span>
  )}
</label>

        </div>
      </section>

    </>
  );
}

export default Dashboard;
