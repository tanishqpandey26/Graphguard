import React from "react";
import ImageSection from "@/components/sections/ImageSection";

const images = 
[
  { name: 'confusion_matrix', alt: 'Confusion Matrix', extension: 'png' },
]


function ProjectDocs() {
  return (
    <>
      <ImageSection images={images} />
    </>
  );
}

export default ProjectDocs;