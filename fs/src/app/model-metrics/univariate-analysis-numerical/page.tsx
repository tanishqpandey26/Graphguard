import React from "react";
import ImageSection from "@/components/sections/ImageSection";

const images =
[
  { name: 'graph_1_n', alt: 'This histogram shows that most transaction amounts are heavily skewed to the left, with a large number of small transactions and very few high-value ones, indicating right-skewed distribution.', extension: 'png' },
  { name: 'graph_2_n', alt: 'Applying a log transformation smooths the right-skewed distribution, revealing a clearer bell-shaped curve. This transformation makes patterns more visible and supports better modeling and visualization for ML tasks.', extension: 'png' },
  { name: 'graph_3_n', alt: 'The histogram shows a distribution of transactions across ZIP codes, indicating that certain areas have significantly higher billing activity. This could reflect population density, customer concentration, or regional spending behavior.', extension: 'png' },
  { name: 'graph_4_n', alt: 'The histogram reveals that the majority of transactions occur in smaller cities, with fewer transactions in highly populated areas. This skewed distribution implies a heavy concentration of users in lower density regions, suggesting urban-rural transaction activity spread.', extension: 'png' },
  { name: 'graph_5_n', alt: 'After applying a log transformation, the population distribution becomes more normalized. This helps in reducing skewness and allows better statistical analysis. It reveals clearer insights into mid sized cities, which were previously masked by extreme values.', extension: 'png' },
  { name: 'graph_6_n', alt: 'The UNIX time distribution shows a higher transaction density in recent periods. This implies increasing transaction activity over time, likely due to digital adoption or user base growth.', extension: 'png' },
  { name: 'graph_7_n', alt: 'The histogram shows a distribution of transactions across years, helps identify if transactions increased or decreased over time.', extension: 'png' },
  { name: 'graph_8_n', alt: 'The pie chart reinforces the bar plot, highlighting that the bulk of transactions occurred in recent years.', extension: 'png' },
  { name: 'graph_9_n', alt: 'Transactions peak mid-year and dip slightly in early and late months. This suggests seasonal behavior, possibly linked to holidays, events, or financial cycles affecting user spending.', extension: 'png' },
  { name: 'graph_10_n', alt: 'This pie chart emphasizes month-wise transaction patterns, showing fairly even distribution with mild surges in specific months, consistent with seasonal spending or promotional periods.', extension: 'png' },
  { name: 'graph_11_n', alt: 'Transaction activity is fairly consistent across days, with some days (like 1st, 15th, or 28th) showing noticeable spikes—likely influenced by salary dates or billing cycles.', extension: 'png' },
  { name: 'graph_12_n', alt: 'Transactions peak between 12 PM–23 PM (~65000 transactions), reflecting typical shopping and bill‐payment windows. Overnight hours and early‐morning (0–11 AM) have minimal activity(~42000 transactions), indicating low consumer engagement and potential fraud monitoring gaps during quieter periods. This underscores a clear midday-to-late-evening transaction peak when consumer activity is highest.', extension: 'png' },
  { name: 'graph_13_n', alt: 'Mondays and Sundays top volume (~25500/25000 transactions), while mid-week (Wednesday ~130000) is slowest. Weekend and start-of-week spikes suggest bill payments and leisure spending drive higher activity at week’s edges.', extension: 'png' },
  { name: 'graph_14_n', alt: 'Most cardholders are aged 25–55, forming a bell-shaped curve peaking around 35–45. Younger (<20) and older (>65) users are scarce. This demographic concentration indicates where marketing and risk models should focus.', extension: 'png' },
  { name: 'graph_15_n', alt: 'Most customers average 20–70 USD per transaction, but a tail extends above 100 USD, indicating a few high-spenders. This variation highlights distinct user segments for targeted fraud thresholds.', extension: 'png' },
  { name: 'graph_16_n', alt: 'Most customers’ single largest purchase caps under 2,000 USD, yet a heavy tail extends beyond 10,000 USD—even up to 30,000 USD —signaling occasional large purchases that warrant elevated fraud scrutiny.', extension: 'png' },
  { name: 'graph_17_n', alt: 'Nearly all customers have minimum transactions under 2 USD, showing many micro-transactions (e.g., subscriptions, small fees). Very few users avoid small purchases, suggesting routine low-value behavior underpins most user profiles.', extension: 'png' },
  { name: 'graph_18_n', alt: 'Most customers exhibit low spending variability (std <150 USD), clustering around 75–125 USD. A long right tail stretches beyond 600 USD, identifying a small segment with highly inconsistent transaction amounts—potentially signaling erratic or fraudulent behavior.', extension: 'png' },
  { name: 'graph_19_n', alt: 'Merchant averages concentrate between 50–70 USD, reflecting common retail price points. A secondary bump around 80–100 USD indicates higher‐end vendors. Very few merchants average above 120 USD, marking them as outliers for specialized or luxury goods. This spread reflects varied business types—helpful for merchant-level risk profiling.', extension: 'png' },
  { name: 'graph_20_n', alt: 'The vast majority of merchants’ highest single sale falls under 2,000 USD, but a pronounced tail extends past 10,000 USD. These rare, large‐ticket transactions highlight high‐value merchants who require intensified fraud monitoring.', extension: 'png' },
  { name: 'graph_21_n', alt: 'Merchant minimums predominantly lie between 0–5 USD, indicating frequent micro-transactions (e.g., convenience stores). Only a few merchants avoid small sales, suggesting niche or high-end services.', extension: 'png' },
  { name: 'graph_22_n', alt: 'Merchant std-dev mostly under 100 USD, showing consistent pricing for many. A smaller group exhibits high variability (> 500 USD), indicating they handle both low- and high-value transactions—key candidates for dynamic fraud thresholds.', extension: 'png' },
  { name: 'graph_23_n', alt: 'Per-capita spend is almost zero for most cities, emphasizing large populations dilute spending. A handful of cities show values >50 USD per resident, suggesting unusually high local or tourist spending needing deeper investigation.', extension: 'png' },
  { name: 'graph_24_n', alt: 'Log-transformation reveals a more continuous distribution peaking around –3 to –2 (i.e. 0.05–0.1 USD per capita), making moderate per-capita spend differences visible and helping pinpoint cities with unexpectedly high normalized outlays.', extension: 'png' },
  { name: 'graph_25_n', alt: 'Distances form a roughly normal shape centered at 70–90 km, showing typical local/regional purchases. The left tail (<10 km) captures neighborhood spending, while extremes (>120 km) flag long-distance or card-not-present transactions warranting extra fraud checks.', extension: 'png' }
]


function ProjectDocs() {
  return (
    <>
      <ImageSection images={images} />
    </>
  );
}

export default ProjectDocs;