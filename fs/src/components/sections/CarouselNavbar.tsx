"use client";

import Link from 'next/link';
import styles from './CarouselNavbar.module.css';

const navLinks = [
  { name: 'Home', path: '/' },
  { name: 'Main', path: '/model-metrics' },
  { name: 'Docs', path: '/docs' },
  { name: 'Univariate-Analysis : Numerical', path: '/model-metrics/univariate-analysis-numerical' },
  { name: 'Univariate-Analysis : Categorical', path: '/model-metrics/univariate-analysis-categorical' },
  { name: 'Confusion Matrix', path: '/model-metrics/confusion-matrix' },
  { name: 'Precision Recall Curve', path: '/model-metrics/precision-recall-curve' },
  { name: 'ROC Curve', path: '/model-metrics/roc-curve' },
  { name: 'Threshold Impact', path: '/model-metrics/threshold-impact' },

];

const CarouselNavbar = () => {
  return (
    <nav className={styles.carouselNav}>
      <div className={styles.navContainer}>
        {navLinks.map((link) => (
          <Link key={link.path} href={link.path} className={styles.navItem}>
            {link.name}
          </Link>
        ))}
      </div>
    </nav>
  );
};

export default CarouselNavbar;
