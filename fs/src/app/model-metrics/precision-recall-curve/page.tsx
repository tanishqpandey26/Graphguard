import React from "react";
import ImageSection from "@/components/sections/ImageSection";

const images = 
[
  { name: 'precision_recall_curve', alt: 'Precision Recall Curve', extension: 'png' },
 ]


function ProjectDocs() {
  return (
    <>
      <ImageSection images={images} />
    </>
  );
}

export default ProjectDocs;