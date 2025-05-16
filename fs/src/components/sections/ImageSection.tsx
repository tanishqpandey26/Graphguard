'use client';

import styles from './ImageSection.module.css';

type ImageData = {
  name: string;
  alt: string;
  extension?: string;
};

type Props = {
  images: ImageData[];
};

const ImageSection = ({ images }: Props) => {
  return (
    <section className={styles.imageSection}>
      {images.map((img, index) => {
        const src = `/images/${img.name}.${img.extension || 'jpg'}`;
        return (
          <div key={index} className={styles.imageWrapper}>
            <img src={src} alt={img.alt} className={styles.image} loading="lazy" />
            <p className={styles.imageText}>{img.alt}</p>
          </div>
        );
      })}
    </section>
  );
};

export default ImageSection;
