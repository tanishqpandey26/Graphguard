import React from "react";
import ImageSection from "@/components/sections/ImageSection";

const images = 
[
  { name: 'roc_curve', alt: 'ROC Curve', extension: 'png' },
]


function ProjectDocs() {
  return (
    <>
      <ImageSection images={images} />
    </>
  );
}

export default ProjectDocs;