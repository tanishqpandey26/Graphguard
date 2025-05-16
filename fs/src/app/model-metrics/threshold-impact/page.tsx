import React from "react";
import ImageSection from "@/components/sections/ImageSection";

const images = 
[
  { name: 'threshold_impact', alt: 'Threshold Impact', extension: 'png' },
]


function ProjectDocs() {
  return (
    <>
      <ImageSection images={images} />
    </>
  );
}

export default ProjectDocs;