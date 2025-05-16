import React from "react";
import ImageSection from "@/components/sections/ImageSection";

const images = 
[
  { name: 'graph_1-c', alt: 'This bar plot shows how transactions are distributed across different spending categories. Categories like gas_transport and grocery_pos are the most frequently occurring. A high count in certain categories might indicate where fraud could be more concentrated.', extension: 'png' },
  { name: 'graph_2-c', alt: 'The pie chart gives a percentage view of how transactions are divided across categories. This helps visually grasp which types dominate overall spending behavior.', extension: 'png' },
  { name: 'graph_3-c', alt: 'This bar plot shows that females make slightly more transactions than males in the dataset. It helps highlight which gender is more active in terms of transaction volume.', extension: 'png' },
  { name: 'graph_4-c', alt: 'The pie chart reinforces the bar plot, showing that the distribution of transactions is relatively balanced, with a slight majority from female users, indicating no strong gender bias in overall activity.', extension: 'png' },
  { name: 'graph_5-c', alt: 'This plot highlights the top 10 cities by transaction volume, revealing that a few urban centers dominate activity. These cities likely represent economic hubs with higher digital transaction engagement.', extension: 'png' },
  { name: 'graph_6-c', alt: 'These cities have the least customer activity, indicating they are either sparsely populated, less commercially active, or underrepresented in the dataset. Fraud risk here may be low due to fewer transactions, but anomalies may go unnoticed.', extension: 'png' },
  { name: 'graph_7-c', alt: 'States with the highest transaction counts likely have larger populations or greater commercial activity. These states are important to monitor for both business performance and fraud trends.', extension: 'png' },
  { name: 'graph_8-c', alt: 'These states have the least transaction activity, suggesting lower customer penetration or spending. While fraud is likely rare, any anomalies may stand out significantly in such small samples.', extension: 'png' },
  { name: 'graph_9-c', alt: 'Film/video editors lead in transactions, followed by exhibition designers and naval architects. Creative, technical, and financial job roles dominate, suggesting higher digital engagement and spending activity among professionals in these fields.', extension: 'png' },
  { name: 'graph_10-c', alt: 'Jobs like hydrogeologist, structural engineer, and tour manager show the fewest transactions, indicating either niche representation, lower income levels, or limited digital spending activity in these professions compared to others.', extension: 'png' },
  { name: 'graph_11-c', alt: 'Large cities record the highest transaction count, slightly ahead of small and medium cities. This suggests strong digital transaction adoption across all city tiers, with urban areas marginally leading due to greater infrastructure and tech penetration.', extension: 'png' },
  { name: 'graph_12-c', alt: 'Most transactions occur in the afternoon and evening, consistent with shopping and bill payment behavior. Fraudsters may also exploit these busy hours to blend into normal patterns.', extension: 'png' },
  { name: 'graph_13-c', alt: 'Spikes in spring and winter likely reflect holiday, shopping and travel spending. Summer and Autumn are comparatively moderate, helping identify seasonal spending and fraud risk patterns.', extension: 'png' },
  { name: 'graph_14-c', alt: 'Adults and middle-aged individuals perform the most transactions, being in their peak earning/spending years. Teenagers and seniors show reduced activity, possibly due to limited financial autonomy or digital literacy.', extension: 'png' }
]


function ProjectDocs() {
  return (
    <>
      <ImageSection images={images} />
    </>
  );
}

export default ProjectDocs;